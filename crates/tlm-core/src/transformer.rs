//! A GPT-style transformer with a hand-written forward AND backward pass.
//!
//!   ids -> token emb + position emb
//!       -> N x Block:   x += MultiHeadAttention(LayerNorm(x))
//!                       x += FeedForward(LayerNorm(x))
//!       -> LayerNorm -> output layer -> logits (one score per vocab entry)
//!
//! All parameters live in one flat `Vec<Matrix>` so the optimizer and the
//! gradient checker can treat them uniformly. See the index constants below.

use crate::attention::causal_softmax_rows;
use crate::mlp::{add_bias, random_matrix, sum_rows};
use crate::ops::{sample_from_probs, softmax, softmax_rows};
use crate::{profile, Matrix, Rng};

#[derive(Debug, Clone, Copy)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    /// Maximum sequence length (number of position embeddings).
    pub ctx: usize,
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
}

/// Put this in a target slot to say "do not learn from this position":
/// no loss is counted and no gradient flows. Used for prompts and padding.
pub const IGNORE_TARGET: usize = usize::MAX;

// ---- parameter layout -------------------------------------------------------
const TOK_EMB: usize = 0; // vocab x d
const POS_EMB: usize = 1; // ctx x d
const FIRST_LAYER: usize = 2;
const PER_LAYER: usize = 13;
// Offsets inside one layer's block of 13 parameters:
const LN1_G: usize = 0; // 1 x d   LayerNorm gain
const LN1_B: usize = 1; // 1 x d   LayerNorm shift
const WQ: usize = 2; //     d x d   (all heads side by side)
const WK: usize = 3;
const WV: usize = 4;
const WO: usize = 5; //     d x d   mixes the concatenated heads
const BO: usize = 6; //     1 x d
const LN2_G: usize = 7;
const LN2_B: usize = 8;
const W1: usize = 9; //     d x 4d  feed-forward, expand
const B1: usize = 10; //    1 x 4d
const W2: usize = 11; //    4d x d  feed-forward, shrink
const B2: usize = 12; //    1 x d
// After the layers: final LN gain, final LN shift, output layer (d x vocab).

pub struct Transformer {
    pub cfg: TransformerConfig,
    pub params: Vec<Matrix>,
}

/// What attention did on one sequence of length T (see `Transformer::attention_trace`).
/// Both lists are indexed `[layer][head]`, and every matrix is T x T with row i = the position
/// that is "looking" and column j = the position it looks at.
pub struct AttentionTrace {
    /// Attention weights: row i spreads 1.0 over positions 0..=i; columns right of i are 0.
    pub weights: Vec<Vec<Matrix>>,
    /// Raw scores `q_i . k_j / sqrt(d_head)` BEFORE the softmax. Entries right of the diagonal are
    /// computed but never used, because a token may not look at the future.
    pub scores: Vec<Vec<Matrix>>,
}

/// Human-readable name of every parameter, in `params` order. Used as the
/// tensor names in saved files, so other tools can understand them.
pub fn param_names(cfg: &TransformerConfig) -> Vec<String> {
    let mut names = vec!["tok_emb".to_string(), "pos_emb".to_string()];
    for l in 0..cfg.n_layer {
        for part in [
            "ln1.gain", "ln1.shift", "attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.bo",
            "ln2.gain", "ln2.shift", "mlp.w1", "mlp.b1", "mlp.w2", "mlp.b2",
        ] {
            names.push(format!("layers.{l}.{part}"));
        }
    }
    names.extend(["lnf.gain", "lnf.shift", "head"].map(String::from));
    names
}

/// (rows, cols) of every parameter, in `params` order, computed without allocating.
pub fn param_shapes(cfg: &TransformerConfig) -> Vec<(usize, usize)> {
    let d = cfg.d_model;
    let mut shapes = vec![(cfg.vocab_size, d), (cfg.ctx, d)];
    for _ in 0..cfg.n_layer {
        shapes.extend([
            (1, d), (1, d), (d, d), (d, d), (d, d), (d, d), (1, d),
            (1, d), (1, d), (d, d.saturating_mul(4)), (1, d.saturating_mul(4)),
            (d.saturating_mul(4), d), (1, d),
        ]);
    }
    shapes.extend([(1, d), (1, d), (d, cfg.vocab_size)]);
    shapes
}

// ---- small building blocks ----------------------------------------------------

/// What LayerNorm must remember for its backward pass.
struct LnCache {
    xhat: Matrix,   // normalised input (mean 0, variance 1 per row)
    rstd: Vec<f32>, // 1 / sqrt(variance + eps), per row
}

/// y = xhat * gain + shift, where xhat = (x - mean) / std, row by row.
fn ln_forward(x: &Matrix, gain: &Matrix, shift: &Matrix) -> (Matrix, LnCache) {
    let _t = profile::span("layernorm fwd");
    let (t, d) = (x.rows, x.cols);
    let mut xhat = Matrix::zeros(t, d);
    let mut y = Matrix::zeros(t, d);
    let mut rstd = vec![0.0; t];
    for r in 0..t {
        let row = &x.data[r * d..(r + 1) * d];
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let rs = 1.0 / (var + 1e-5).sqrt();
        rstd[r] = rs;
        for j in 0..d {
            let xh = (row[j] - mean) * rs;
            xhat.data[r * d + j] = xh;
            y.data[r * d + j] = xh * gain.data[j] + shift.data[j];
        }
    }
    (y, LnCache { xhat, rstd })
}

/// Returns (dx, dgain, dshift).
/// With dxhat = dy * gain:   dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
fn ln_backward(c: &LnCache, dy: &Matrix, gain: &Matrix) -> (Matrix, Matrix, Matrix) {
    let _t = profile::span("layernorm bwd");
    let (t, d) = (dy.rows, dy.cols);
    let mut dx = Matrix::zeros(t, d);
    let mut dgain = Matrix::zeros(1, d);
    let mut dshift = Matrix::zeros(1, d);
    for r in 0..t {
        let (mut m1, mut m2) = (0.0, 0.0);
        for j in 0..d {
            let dyj = dy.data[r * d + j];
            let xh = c.xhat.data[r * d + j];
            dgain.data[j] += dyj * xh;
            dshift.data[j] += dyj;
            let dxh = dyj * gain.data[j];
            m1 += dxh;
            m2 += dxh * xh;
        }
        m1 /= d as f32;
        m2 /= d as f32;
        for j in 0..d {
            let xh = c.xhat.data[r * d + j];
            let dxh = dy.data[r * d + j] * gain.data[j];
            dx.data[r * d + j] = c.rstd[r] * (dxh - m1 - xh * m2);
        }
    }
    (dx, dgain, dshift)
}

const SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// tanh as a ratio of two polynomials (the float approximation used by the
/// Eigen library, accurate to about 1e-7). Unlike `f32::tanh`, which calls a
/// scalar library routine, this is only +, * and /, so the compiler can
/// vectorize loops over it and process several numbers per instruction.
#[inline(always)]
fn fast_tanh(x: f32) -> f32 {
    let x = x.clamp(-7.905_311, 7.905_311); // beyond this tanh is 1 to float precision
    let x2 = x * x;
    let p = x
        * (4.893_524_6e-3
            + x2 * (6.372_619_3e-4
                + x2 * (1.485_722_4e-5
                    + x2 * (5.122_297e-8 + x2 * (-8.604_672e-11 + x2 * (2.000_188e-13 + x2 * -2.760_768_5e-16))))));
    let q = 4.893_525e-3 + x2 * (2.268_434_6e-3 + x2 * (1.185_347e-4 + x2 * 1.198_258_4e-6));
    p / q
}

/// GELU, a smooth cousin of ReLU (tanh approximation, as in GPT-2).
#[inline(always)]
fn gelu(x: f32) -> f32 {
    let u = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + fast_tanh(u))
}

#[inline(always)]
fn gelu_grad(x: f32) -> f32 {
    let u = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
    let t = fast_tanh(u);
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * x * x)
}

/// Copy columns c0..c1 of `m` into a new matrix (used to carve out one head).
fn cols(m: &Matrix, c0: usize, c1: usize) -> Matrix {
    let w = c1 - c0;
    let mut out = Matrix::zeros(m.rows, w);
    for r in 0..m.rows {
        out.data[r * w..(r + 1) * w].copy_from_slice(&m.data[r * m.cols + c0..r * m.cols + c1]);
    }
    out
}

/// Write `src` into `dst` starting at column c0 (used to glue heads back together).
fn put_cols(dst: &mut Matrix, c0: usize, src: &Matrix) {
    for r in 0..src.rows {
        let start = r * dst.cols + c0;
        dst.data[start..start + src.cols].copy_from_slice(&src.data[r * src.cols..(r + 1) * src.cols]);
    }
}

fn acc(dst: &mut Matrix, src: &Matrix) {
    let _t = profile::span("grad accumulate");
    for (a, b) in dst.data.iter_mut().zip(&src.data) {
        *a += b;
    }
}

/// Multi-head causal attention on already-projected Q, K, V (each T x d).
/// Head h uses columns h*dh .. (h+1)*dh. Returns (concatenated output, per-head weights).
fn mha_forward(q: &Matrix, k: &Matrix, v: &Matrix, n_head: usize) -> (Matrix, Vec<Matrix>) {
    let _t = profile::span("attention core fwd");
    let (t, d) = (q.rows, q.cols);
    let dh = d / n_head;
    let scale = 1.0 / (dh as f32).sqrt();
    let mut out = Matrix::zeros(t, d);
    let mut weights = Vec::with_capacity(n_head);
    for h in 0..n_head {
        let (c0, c1) = (h * dh, (h + 1) * dh);
        let mut s = cols(q, c0, c1).matmul(&cols(k, c0, c1).transpose());
        for x in s.data.iter_mut() {
            *x *= scale;
        }
        let a = causal_softmax_rows(&s);
        put_cols(&mut out, c0, &a.matmul(&cols(v, c0, c1)));
        weights.push(a);
    }
    (out, weights)
}

/// Same math as the single-head backward in attention.rs, once per head.
fn mha_backward(
    dout: &Matrix,
    q: &Matrix,
    k: &Matrix,
    v: &Matrix,
    weights: &[Matrix],
    n_head: usize,
) -> (Matrix, Matrix, Matrix) {
    let _t = profile::span("attention core bwd");
    let (t, d) = (q.rows, q.cols);
    let dh = d / n_head;
    let scale = 1.0 / (dh as f32).sqrt();
    let (mut dq, mut dk, mut dv) = (Matrix::zeros(t, d), Matrix::zeros(t, d), Matrix::zeros(t, d));
    for h in 0..n_head {
        let (c0, c1) = (h * dh, (h + 1) * dh);
        let (qh, kh, vh) = (cols(q, c0, c1), cols(k, c0, c1), cols(v, c0, c1));
        let doh = cols(dout, c0, c1);
        let a = &weights[h];

        let da = doh.matmul(&vh.transpose());
        let dvh = a.matmul_at_b(&doh);

        let mut ds = Matrix::zeros(t, t);
        for i in 0..t {
            let row = i * t..(i + 1) * t;
            let dot: f32 = a.data[row.clone()].iter().zip(&da.data[row.clone()]).map(|(x, g)| x * g).sum();
            for j in 0..t {
                ds.data[i * t + j] = a.data[i * t + j] * (da.data[i * t + j] - dot);
            }
        }
        let mut dqh = ds.matmul(&kh);
        let mut dkh = ds.matmul_at_b(&qh);
        for x in dqh.data.iter_mut().chain(dkh.data.iter_mut()) {
            *x *= scale;
        }
        put_cols(&mut dq, c0, &dqh);
        put_cols(&mut dk, c0, &dkh);
        put_cols(&mut dv, c0, &dvh);
    }
    (dq, dk, dv)
}

// ---- the model ---------------------------------------------------------------

struct LayerCache {
    ln1: LnCache,
    ln1_out: Matrix,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    attn_weights: Vec<Matrix>,
    o_cat: Matrix,
    ln2: LnCache,
    ln2_out: Matrix,
    hpre: Matrix, // feed-forward hidden layer before GELU
    hact: Matrix, // ... after GELU
}

struct FwdCache {
    layers: Vec<LayerCache>,
    lnf: LnCache,
    lnf_out: Matrix,
    logits: Matrix,
}

impl Transformer {
    pub fn new(cfg: TransformerConfig, rng: &mut Rng) -> Self {
        assert_eq!(cfg.d_model % cfg.n_head, 0, "d_model must divide evenly into heads");
        let d = cfg.d_model;
        let ones = || Matrix::from_vec(1, d, vec![1.0; d]);
        let mut params = vec![random_matrix(cfg.vocab_size, d, d, rng), random_matrix(cfg.ctx, d, d, rng)];
        for _ in 0..cfg.n_layer {
            params.extend([
                ones(),                           // LN1_G
                Matrix::zeros(1, d),              // LN1_B
                random_matrix(d, d, d, rng),      // WQ
                random_matrix(d, d, d, rng),      // WK
                random_matrix(d, d, d, rng),      // WV
                random_matrix(d, d, d, rng),      // WO
                Matrix::zeros(1, d),              // BO
                ones(),                           // LN2_G
                Matrix::zeros(1, d),              // LN2_B
                random_matrix(d, 4 * d, d, rng),  // W1
                Matrix::zeros(1, 4 * d),          // B1
                random_matrix(4 * d, d, 4 * d, rng), // W2
                Matrix::zeros(1, d),              // B2
            ]);
        }
        params.extend([ones(), Matrix::zeros(1, d), random_matrix(d, cfg.vocab_size, d, rng)]);
        Transformer { cfg, params }
    }

    /// Rebuild a model from loaded weights, checking they fit the config.
    pub fn from_params(cfg: TransformerConfig, params: Vec<Matrix>) -> Result<Self, String> {
        if cfg.n_head == 0 || cfg.d_model == 0 || cfg.d_model % cfg.n_head != 0 {
            return Err("invalid config: d_model must be a positive multiple of n_head".into());
        }
        let shapes = param_shapes(&cfg);
        if params.len() != shapes.len() {
            return Err(format!("expected {} tensors, found {}", shapes.len(), params.len()));
        }
        for (i, (p, &(r, c))) in params.iter().zip(&shapes).enumerate() {
            if (p.rows, p.cols) != (r, c) {
                return Err(format!("tensor {i} has shape {}x{}, expected {r}x{c}", p.rows, p.cols));
            }
        }
        Ok(Transformer { cfg, params })
    }

    pub fn num_params(&self) -> usize {
        self.params.iter().map(|p| p.data.len()).sum()
    }

    pub fn zero_grads(&self) -> Vec<Matrix> {
        self.params.iter().map(|p| Matrix::zeros(p.rows, p.cols)).collect()
    }

    /// Parameter `o` of layer `l`.
    fn lp(&self, l: usize, o: usize) -> &Matrix {
        &self.params[FIRST_LAYER + l * PER_LAYER + o]
    }

    fn final_index(&self) -> usize {
        FIRST_LAYER + self.cfg.n_layer * PER_LAYER
    }

    fn forward(&self, ids: &[usize]) -> FwdCache {
        let c = self.cfg;
        let (t, d) = (ids.len(), c.d_model);
        assert!(t > 0 && t <= c.ctx, "sequence length must be 1..=ctx");

        // Embeddings: what the token is + where it sits.
        let (tok, pos) = (&self.params[TOK_EMB], &self.params[POS_EMB]);
        let mut x = Matrix::zeros(t, d);
        for i in 0..t {
            for j in 0..d {
                x.data[i * d + j] = tok.data[ids[i] * d + j] + pos.data[i * d + j];
            }
        }

        let mut layers = Vec::with_capacity(c.n_layer);
        for l in 0..c.n_layer {
            // Attention sub-block, with a residual connection.
            let (ln1_out, ln1) = ln_forward(&x, self.lp(l, LN1_G), self.lp(l, LN1_B));
            let q = ln1_out.matmul(self.lp(l, WQ));
            let k = ln1_out.matmul(self.lp(l, WK));
            let v = ln1_out.matmul(self.lp(l, WV));
            let (o_cat, attn_weights) = mha_forward(&q, &k, &v, c.n_head);
            let mut attn_out = o_cat.matmul(self.lp(l, WO));
            add_bias(&mut attn_out, self.lp(l, BO));
            let x_mid = x.add(&attn_out);

            // Feed-forward sub-block, with a residual connection.
            let (ln2_out, ln2) = ln_forward(&x_mid, self.lp(l, LN2_G), self.lp(l, LN2_B));
            let mut hpre = ln2_out.matmul(self.lp(l, W1));
            add_bias(&mut hpre, self.lp(l, B1));
            let mut hact = hpre.clone();
            let gelu_t = profile::span("gelu fwd");
            for z in hact.data.iter_mut() {
                *z = gelu(*z);
            }
            drop(gelu_t);
            let mut mlp_out = hact.matmul(self.lp(l, W2));
            add_bias(&mut mlp_out, self.lp(l, B2));
            x = x_mid.add(&mlp_out);

            layers.push(LayerCache { ln1, ln1_out, q, k, v, attn_weights, o_cat, ln2, ln2_out, hpre, hact });
        }

        let f = self.final_index();
        let (lnf_out, lnf) = ln_forward(&x, &self.params[f], &self.params[f + 1]);
        let logits = lnf_out.matmul(&self.params[f + 2]);
        FwdCache { layers, lnf, lnf_out, logits }
    }

    /// Scores for the next token at every position (T x vocab).
    pub fn logits(&self, ids: &[usize]) -> Matrix {
        self.forward(ids).logits
    }

    /// Look inside attention: run the model on `ids` and report, for every layer and head, how
    /// each position scored and weighted the positions before it. For teaching and inspection;
    /// it does the same forward pass as `logits` plus a little extra work to recover the scores.
    pub fn attention_trace(&self, ids: &[usize]) -> AttentionTrace {
        let cache = self.forward(ids);
        let dh = self.cfg.d_model / self.cfg.n_head;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut weights = Vec::with_capacity(self.cfg.n_layer);
        let mut scores = Vec::with_capacity(self.cfg.n_layer);
        for lc in &cache.layers {
            weights.push(lc.attn_weights.clone());
            let per_head: Vec<Matrix> = (0..self.cfg.n_head)
                .map(|h| {
                    let mut s = cols(&lc.q, h * dh, (h + 1) * dh).matmul(&cols(&lc.k, h * dh, (h + 1) * dh).transpose());
                    s.data.iter_mut().for_each(|x| *x *= scale);
                    s
                })
                .collect();
            scores.push(per_head);
        }
        AttentionTrace { weights, scores }
    }

    /// Transposed copies of the weights the backward pass multiplies by.
    /// Weights do not change within a step, so we build these ONCE per batch
    /// and share them, instead of re-transposing for every sequence.
    /// Indexed like `params`; entries that are never needed are empty.
    fn transposed_weights(&self) -> Vec<Matrix> {
        let _t = profile::span("weight transposes");
        let mut wt: Vec<Matrix> = (0..self.params.len()).map(|_| Matrix::zeros(0, 0)).collect();
        for l in 0..self.cfg.n_layer {
            for offset in [WQ, WK, WV, WO, W1, W2] {
                let i = FIRST_LAYER + l * PER_LAYER + offset;
                wt[i] = self.params[i].transpose();
            }
        }
        let head = self.final_index() + 2;
        wt[head] = self.params[head].transpose();
        wt
    }

    /// Forward + backward for ONE sequence. Adds into `grads` and returns the
    /// SUM of per-token losses. `inv_n` = 1 / (total tokens in the batch).
    /// `wt` comes from `transposed_weights`.
    fn sequence_pass(&self, ids: &[usize], targets: &[usize], inv_n: f32, wt: &[Matrix], grads: &mut [Matrix]) -> f32 {
        let c = self.cfg;
        let (t, d, v) = (ids.len(), c.d_model, c.vocab_size);
        let fwd_t = profile::span("forward");
        let cache = self.forward(ids);
        drop(fwd_t);

        // Loss and its gradient at the scores: (probs - one_hot) / N.
        let loss_t = profile::span("loss softmax");
        let probs = softmax_rows(&cache.logits);
        let mut loss_sum = 0.0;
        let mut dlogits = probs.clone();
        for (i, &tgt) in targets.iter().enumerate() {
            if tgt == IGNORE_TARGET {
                // Masked position (prompt or padding): no loss, and no gradient at all.
                dlogits.data[i * v..(i + 1) * v].fill(0.0);
                continue;
            }
            loss_sum += -probs.get(i, tgt).ln();
            dlogits.data[i * v + tgt] -= 1.0;
        }
        for g in dlogits.data.iter_mut() {
            *g *= inv_n;
        }
        drop(loss_t);

        // Output layer and final LayerNorm.
        let _bwd_t = profile::span("backward");
        let f = self.final_index();
        acc(&mut grads[f + 2], &cache.lnf_out.matmul_at_b(&dlogits));
        let dlnf_out = dlogits.matmul(&wt[f + 2]);
        let (mut dx, dg, db) = ln_backward(&cache.lnf, &dlnf_out, &self.params[f]);
        acc(&mut grads[f], &dg);
        acc(&mut grads[f + 1], &db);

        // Blocks, last to first. `dx` is the gradient flowing down the residual stream.
        for l in (0..c.n_layer).rev() {
            let lc = &cache.layers[l];
            let base = FIRST_LAYER + l * PER_LAYER;

            // Feed-forward sub-block (x_out = x_mid + mlp_out, so dx flows to both).
            acc(&mut grads[base + W2], &lc.hact.matmul_at_b(&dx));
            acc(&mut grads[base + B2], &sum_rows(&dx));
            let mut dh = dx.matmul(&wt[base + W2]);
            let gelu_t = profile::span("gelu bwd");
            for (g, &pre) in dh.data.iter_mut().zip(&lc.hpre.data) {
                *g *= gelu_grad(pre);
            }
            drop(gelu_t);
            acc(&mut grads[base + W1], &lc.ln2_out.matmul_at_b(&dh));
            acc(&mut grads[base + B1], &sum_rows(&dh));
            let dln2_out = dh.matmul(&wt[base + W1]);
            let (dxl, dg, db) = ln_backward(&lc.ln2, &dln2_out, self.lp(l, LN2_G));
            acc(&mut grads[base + LN2_G], &dg);
            acc(&mut grads[base + LN2_B], &db);
            let dx_mid = dx.add(&dxl);

            // Attention sub-block.
            acc(&mut grads[base + WO], &lc.o_cat.matmul_at_b(&dx_mid));
            acc(&mut grads[base + BO], &sum_rows(&dx_mid));
            let do_cat = dx_mid.matmul(&wt[base + WO]);
            let (dq, dk, dv) = mha_backward(&do_cat, &lc.q, &lc.k, &lc.v, &lc.attn_weights, c.n_head);
            acc(&mut grads[base + WQ], &lc.ln1_out.matmul_at_b(&dq));
            acc(&mut grads[base + WK], &lc.ln1_out.matmul_at_b(&dk));
            acc(&mut grads[base + WV], &lc.ln1_out.matmul_at_b(&dv));
            let dln1_out = dq
                .matmul(&wt[base + WQ])
                .add(&dk.matmul(&wt[base + WK]))
                .add(&dv.matmul(&wt[base + WV]));
            let (dxl, dg, db) = ln_backward(&lc.ln1, &dln1_out, self.lp(l, LN1_G));
            acc(&mut grads[base + LN1_G], &dg);
            acc(&mut grads[base + LN1_B], &db);
            dx = dx_mid.add(&dxl);
        }

        // Embeddings: position i gets dx[i]; the token id seen there also gets it.
        for i in 0..t {
            for j in 0..d {
                grads[POS_EMB].data[i * d + j] += dx.data[i * d + j];
                grads[TOK_EMB].data[ids[i] * d + j] += dx.data[i * d + j];
            }
        }
        loss_sum
    }

    /// Mean loss and gradients for a batch. `inputs` and `targets` each hold
    /// `n_seq` sequences of `seq_len` tokens, back to back. Sequences are
    /// independent, so they are split across `threads` CPU threads.
    pub fn loss_and_grads(
        &self,
        inputs: &[usize],
        targets: &[usize],
        seq_len: usize,
        threads: usize,
    ) -> (f32, Vec<Matrix>) {
        // Average over the tokens that COUNT (masked ones are excluded).
        let counted = targets.iter().filter(|&&t| t != IGNORE_TARGET).count();
        self.loss_and_grads_scaled(inputs, targets, seq_len, threads, counted)
    }

    /// Like `loss_and_grads`, but divides by `total_counted` instead of by the
    /// number of counted tokens in THIS call. That is what data-parallel
    /// training needs: several workers each process a share of one batch, each
    /// divides by the whole batch's count, and then the workers' results simply
    /// ADD UP to the gradient of the whole batch.
    pub fn loss_and_grads_scaled(
        &self,
        inputs: &[usize],
        targets: &[usize],
        seq_len: usize,
        threads: usize,
        total_counted: usize,
    ) -> (f32, Vec<Matrix>) {
        assert_eq!(inputs.len(), targets.len());
        assert_eq!(inputs.len() % seq_len, 0);
        let n_seq = inputs.len() / seq_len;
        if total_counted == 0 || n_seq == 0 {
            return (0.0, self.zero_grads());
        }
        let inv_n = 1.0 / total_counted as f32;
        self.grads_with_scales(inputs, targets, seq_len, threads, &|_| inv_n)
    }

    /// Gradient of  `sum over sequences s of  scales[s] * NLL_s`,  where `NLL_s` is the
    /// summed negative log-likelihood of sequence s's counted (unmasked) tokens.
    /// A scale may be NEGATIVE: that pushes the model AWAY from the sequence.
    /// This is the building block of preference training (see `dpo`).
    pub fn weighted_grads(&self, inputs: &[usize], targets: &[usize], seq_len: usize, scales: &[f32], threads: usize) -> Vec<Matrix> {
        assert_eq!(inputs.len(), targets.len());
        assert_eq!(inputs.len(), scales.len() * seq_len, "need exactly one scale per sequence");
        if scales.is_empty() {
            return self.zero_grads();
        }
        self.grads_with_scales(inputs, targets, seq_len, threads, &|s| scales[s]).1
    }

    /// Shared engine: forward + backward over a batch, each sequence's gradient
    /// multiplied by `scale_of(sequence index)`. Returns (sum of scale * loss, gradients).
    fn grads_with_scales(
        &self,
        inputs: &[usize],
        targets: &[usize],
        seq_len: usize,
        threads: usize,
        scale_of: &(dyn Fn(usize) -> f32 + Sync),
    ) -> (f32, Vec<Matrix>) {
        let n_seq = inputs.len() / seq_len;
        let wt = self.transposed_weights();
        let work = |first: usize, last: usize| -> (f32, Vec<Matrix>) {
            let mut grads = self.zero_grads();
            let mut loss = 0.0;
            for s in first..last {
                let r = s * seq_len..(s + 1) * seq_len;
                let scale = scale_of(s);
                loss += self.sequence_pass(&inputs[r.clone()], &targets[r], scale, &wt, &mut grads) * scale;
            }
            (loss, grads)
        };

        // Browsers (wasm32) cannot spawn threads the way native code does.
        let threads = if cfg!(target_arch = "wasm32") { 1 } else { threads.clamp(1, n_seq) };
        if threads == 1 {
            return work(0, n_seq);
        }
        let chunk = n_seq.div_ceil(threads);
        let work = &work;
        let results: Vec<(f32, Vec<Matrix>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|i| scope.spawn(move || work((i * chunk).min(n_seq), ((i + 1) * chunk).min(n_seq))))
                .collect();
            handles.into_iter().map(|h| h.join().expect("worker thread panicked")).collect()
        });

        let mut iter = results.into_iter();
        let (mut loss, mut grads) = iter.next().unwrap();
        for (l, g) in iter {
            loss += l;
            for (dst, src) in grads.iter_mut().zip(&g) {
                acc(dst, src);
            }
        }
        (loss, grads)
    }

    /// For each sequence, the SUM of log-probabilities of its counted (unmasked) targets:
    /// "how likely does the model find this response". Computed with a numerically
    /// stable log-softmax, since probabilities this small can underflow to zero.
    pub fn sequence_logprobs(&self, inputs: &[usize], targets: &[usize], seq_len: usize, threads: usize) -> Vec<f32> {
        assert_eq!(inputs.len(), targets.len());
        assert_eq!(inputs.len() % seq_len, 0);
        let (v, n_seq) = (self.cfg.vocab_size, inputs.len() / seq_len);
        let one = |s: usize| -> f32 {
            let r = s * seq_len..(s + 1) * seq_len;
            let logits = self.forward(&inputs[r.clone()]).logits;
            let mut total = 0.0;
            for (i, &tgt) in targets[r].iter().enumerate() {
                if tgt == IGNORE_TARGET {
                    continue;
                }
                let row = &logits.data[i * v..(i + 1) * v];
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let lse = max + row.iter().map(|z| (z - max).exp()).sum::<f32>().ln();
                total += row[tgt] - lse;
            }
            total
        };
        let threads = if cfg!(target_arch = "wasm32") { 1 } else { threads.clamp(1, n_seq.max(1)) };
        if threads == 1 || n_seq < 2 {
            return (0..n_seq).map(one).collect();
        }
        let chunk = n_seq.div_ceil(threads);
        let one = &one;
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|i| scope.spawn(move || ((i * chunk).min(n_seq)..((i + 1) * chunk).min(n_seq)).map(one).collect::<Vec<f32>>()))
                .collect();
            handles.into_iter().flat_map(|h| h.join().expect("worker thread panicked")).collect()
        })
    }

    /// Mean loss only (no gradients). For validation.
    pub fn loss(&self, inputs: &[usize], targets: &[usize], seq_len: usize) -> f32 {
        let v = self.cfg.vocab_size;
        let (mut total, mut counted) = (0.0, 0usize);
        for s in 0..inputs.len() / seq_len {
            let r = s * seq_len..(s + 1) * seq_len;
            let probs = softmax_rows(&self.forward(&inputs[r.clone()]).logits);
            for (i, &tgt) in targets[r].iter().enumerate() {
                if tgt == IGNORE_TARGET {
                    continue;
                }
                debug_assert!(tgt < v);
                total += -probs.get(i, tgt).ln();
                counted += 1;
            }
        }
        if counted == 0 { 0.0 } else { total / counted as f32 }
    }

    /// Generate `len` tokens after `prompt`. Lower temperature = safer, more
    /// repetitive; higher = more surprising. Only the last `ctx` tokens are used.
    pub fn sample(&self, prompt: &[usize], len: usize, temperature: f32, rng: &mut Rng) -> Vec<usize> {
        self.sample_until(prompt, len, temperature, rng, |_| false)
    }

    /// Like `sample`, but after each new token calls `stop` with everything
    /// generated so far, and ends early when it returns true.
    pub fn sample_until(
        &self,
        prompt: &[usize],
        len: usize,
        temperature: f32,
        rng: &mut Rng,
        mut stop: impl FnMut(&[usize]) -> bool,
    ) -> Vec<usize> {
        assert!(!prompt.is_empty());
        let v = self.cfg.vocab_size;
        let mut history = prompt.to_vec();
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            let start = history.len().saturating_sub(self.cfg.ctx);
            let window = &history[start..];
            let logits = self.forward(window).logits;
            let last = &logits.data[(window.len() - 1) * v..window.len() * v];
            let scaled: Vec<f32> = last.iter().map(|z| z / temperature).collect();
            let next = sample_from_probs(&softmax(&scaled), rng);
            history.push(next);
            out.push(next);
            if stop(&out) {
                break;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::AdamW;

    #[test]
    fn attention_trace_is_a_causal_probability_table_that_matches_its_scores() {
        let cfg = TransformerConfig { vocab_size: 9, ctx: 6, d_model: 8, n_layer: 2, n_head: 2 };
        let model = Transformer::new(cfg, &mut Rng::new(5));
        let ids = [3usize, 1, 4, 1, 5];
        let t = ids.len();
        let trace = model.attention_trace(&ids);
        assert_eq!((trace.weights.len(), trace.weights[0].len()), (2, 2));
        for l in 0..2 {
            for h in 0..2 {
                let (w, s) = (&trace.weights[l][h], &trace.scores[l][h]);
                assert_eq!((w.rows, w.cols, s.rows, s.cols), (t, t, t, t));
                for i in 0..t {
                    let sum: f32 = (0..=i).map(|j| w.get(i, j)).sum();
                    assert!((sum - 1.0).abs() < 1e-5, "row {i} sums to {sum}");
                    for j in i + 1..t {
                        assert_eq!(w.get(i, j), 0.0, "no peeking at the future");
                    }
                    // the weights are exactly softmax of the visible scores
                    let probs = softmax(&(0..=i).map(|j| s.get(i, j)).collect::<Vec<_>>());
                    for (j, p) in probs.iter().enumerate() {
                        assert!((p - w.get(i, j)).abs() < 1e-5);
                    }
                }
            }
        }
        // the first token can only look at itself
        assert_eq!(trace.weights[0][0].get(0, 0), 1.0);
    }

    #[test]
    fn attention_trace_does_not_change_the_model_output() {
        let cfg = TransformerConfig { vocab_size: 9, ctx: 6, d_model: 8, n_layer: 1, n_head: 2 };
        let model = Transformer::new(cfg, &mut Rng::new(6));
        let before = model.logits(&[1, 2, 3]);
        let _ = model.attention_trace(&[1, 2, 3]);
        assert_eq!(before, model.logits(&[1, 2, 3]));
    }

    fn tiny() -> (Transformer, Vec<usize>, Vec<usize>) {
        let cfg = TransformerConfig { vocab_size: 7, ctx: 4, d_model: 8, n_layer: 2, n_head: 2 };
        let mut rng = Rng::new(11);
        let model = Transformer::new(cfg, &mut rng);
        // Two sequences of length 4.
        let inputs = vec![0, 3, 5, 1, 6, 2, 2, 4];
        let targets = vec![3, 5, 1, 6, 2, 2, 4, 0];
        (model, inputs, targets)
    }

    #[test]
    fn masked_positions_are_excluded_from_loss_and_gradients() {
        let (mut model, x, y) = tiny();
        // Keep only the 2nd and 7th targets; mask everything else.
        let mut masked = vec![IGNORE_TARGET; y.len()];
        masked[1] = y[1];
        masked[6] = y[6];

        // The masked loss is the average of just those two per-token losses.
        let only_a: Vec<usize> = (0..y.len()).map(|i| if i == 1 { y[i] } else { IGNORE_TARGET }).collect();
        let only_b: Vec<usize> = (0..y.len()).map(|i| if i == 6 { y[i] } else { IGNORE_TARGET }).collect();
        let expected = (model.loss(&x, &only_a, 4) + model.loss(&x, &only_b, 4)) / 2.0;
        assert!((model.loss(&x, &masked, 4) - expected).abs() < 1e-5);

        // Gradients still match numerical estimates with the mask in place.
        let (loss, grads) = model.loss_and_grads(&x, &masked, 4, 1);
        assert!((loss - expected).abs() < 1e-5);
        let h = 1e-2;
        for p in 0..model.params.len() {
            for idx in (0..model.params[p].data.len()).step_by(3) {
                let original = model.params[p].data[idx];
                model.params[p].data[idx] = original + h;
                let up = model.loss(&x, &masked, 4);
                model.params[p].data[idx] = original - h;
                let down = model.loss(&x, &masked, 4);
                model.params[p].data[idx] = original;
                let numeric = (up - down) / (2.0 * h);
                assert!(
                    (numeric - grads[p].data[idx]).abs() < 2e-3 + 0.05 * grads[p].data[idx].abs(),
                    "param {p} entry {idx}: {numeric} vs {}",
                    grads[p].data[idx]
                );
            }
        }
    }

    #[test]
    fn sequence_logprobs_agree_with_the_loss() {
        let (model, x, y) = tiny();
        let lp = model.sequence_logprobs(&x, &y, 4, 1);
        assert_eq!(lp.len(), 2);
        let mean_nll = -(lp[0] + lp[1]) / 8.0;
        assert!((mean_nll - model.loss(&x, &y, 4)).abs() < 1e-4);
        // Masked targets are skipped, and threads do not change the answer.
        let mut masked = y.clone();
        masked[0] = IGNORE_TARGET;
        let a = model.sequence_logprobs(&x, &masked, 4, 1);
        assert!(a[0] > lp[0], "dropping a (negative) log-prob term must raise the sum");
        assert_eq!(model.sequence_logprobs(&x, &y, 4, 2), lp);
    }

    #[test]
    fn weighted_gradients_match_numerical_estimates_including_negative_weights() {
        // Objective: sum_s scale_s * NLL_s, with one POSITIVE and one NEGATIVE scale.
        let (mut model, x, y) = tiny();
        let scales = [0.7f32, -1.3];
        let objective = |m: &Transformer| -> f32 {
            let lp = m.sequence_logprobs(&x, &y, 4, 1);
            lp.iter().zip(&scales).map(|(l, s)| -l * s).sum()
        };
        let grads = model.weighted_grads(&x, &y, 4, &scales, 1);
        let h = 1e-2;
        for p in 0..model.params.len() {
            for idx in (0..model.params[p].data.len()).step_by(2) {
                let original = model.params[p].data[idx];
                model.params[p].data[idx] = original + h;
                let up = objective(&model);
                model.params[p].data[idx] = original - h;
                let down = objective(&model);
                model.params[p].data[idx] = original;
                let numeric = (up - down) / (2.0 * h);
                assert!(
                    (numeric - grads[p].data[idx]).abs() < 4e-3 + 0.05 * grads[p].data[idx].abs(),
                    "param {p} entry {idx}: numeric {numeric} vs analytic {}",
                    grads[p].data[idx]
                );
            }
        }
        // Threads give the same gradients.
        let again = model.weighted_grads(&x, &y, 4, &scales, 2);
        for (a, b) in grads.iter().zip(&again) {
            for (u, v) in a.data.iter().zip(&b.data) {
                assert!((u - v).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn fully_masked_batch_gives_zero_loss_and_zero_gradients() {
        let (model, x, y) = tiny();
        let all_masked = vec![IGNORE_TARGET; y.len()];
        let (loss, grads) = model.loss_and_grads(&x, &all_masked, 4, 1);
        assert_eq!(loss, 0.0);
        assert!(grads.iter().all(|g| g.data.iter().all(|&v| v == 0.0)));
        assert_eq!(model.loss(&x, &all_masked, 4), 0.0);
    }

    #[test]
    fn sample_until_stops_early() {
        let (model, _, _) = tiny();
        let out = model.sample_until(&[1, 2], 50, 1.0, &mut Rng::new(3), |generated| generated.len() >= 5);
        assert_eq!(out.len(), 5);
        let full = model.sample(&[1, 2], 50, 1.0, &mut Rng::new(3));
        assert_eq!(full.len(), 50);
        assert_eq!(&full[..5], &out[..]); // same random stream, just cut short
    }

    #[test]
    fn fast_tanh_is_accurate_everywhere() {
        let mut worst = 0.0f64;
        let mut x = -12.0f32;
        while x <= 12.0 {
            let err = (fast_tanh(x) as f64 - (x as f64).tanh()).abs();
            worst = worst.max(err);
            x += 0.0005;
        }
        assert!(worst < 1e-5, "worst error {worst}");
        assert_eq!(fast_tanh(0.0), 0.0);
        assert!((fast_tanh(50.0) - 1.0).abs() < 1e-6 && (fast_tanh(-50.0) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn gelu_matches_its_formula_and_its_gradient() {
        for x in [-4.0f32, -1.0, -0.1, 0.0, 0.3, 2.0, 5.0] {
            let exact = 0.5 * x as f64 * (1.0 + (0.7978845608 * (x as f64 + 0.044715 * (x as f64).powi(3))).tanh());
            assert!((gelu(x) as f64 - exact).abs() < 1e-5);
            let h = 1e-2;
            let numeric = (gelu(x + h) - gelu(x - h)) / (2.0 * h);
            assert!((numeric - gelu_grad(x)).abs() < 2e-3, "x={x}: {numeric} vs {}", gelu_grad(x));
        }
    }

    #[test]
    fn names_and_shapes_describe_the_real_parameters() {
        let (model, _, _) = tiny();
        let names = param_names(&model.cfg);
        let shapes = param_shapes(&model.cfg);
        assert_eq!(names.len(), model.params.len());
        for (i, p) in model.params.iter().enumerate() {
            assert_eq!((p.rows, p.cols), shapes[i], "shape of {}", names[i]);
        }
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(unique.len(), names.len(), "names must be unique");
    }

    #[test]
    fn untrained_loss_is_near_ln_vocab() {
        let (model, x, y) = tiny();
        let loss = model.loss(&x, &y, 4);
        assert!((loss - 7.0f32.ln()).abs() < 0.5, "loss {loss}");
    }

    #[test]
    fn loss_paths_agree() {
        let (model, x, y) = tiny();
        let (train_loss, _) = model.loss_and_grads(&x, &y, 4, 1);
        assert!((train_loss - model.loss(&x, &y, 4)).abs() < 1e-5);
    }

    #[test]
    fn every_parameter_gradient_matches_numerical_estimate() {
        let (mut model, x, y) = tiny();
        let (_, grads) = model.loss_and_grads(&x, &y, 4, 1);
        let h = 1e-2;
        let mut checked = 0;
        for p in 0..model.params.len() {
            for idx in 0..model.params[p].data.len() {
                let original = model.params[p].data[idx];
                model.params[p].data[idx] = original + h;
                let up = model.loss(&x, &y, 4);
                model.params[p].data[idx] = original - h;
                let down = model.loss(&x, &y, 4);
                model.params[p].data[idx] = original;
                let numeric = (up - down) / (2.0 * h);
                let analytic = grads[p].data[idx];
                assert!(
                    (numeric - analytic).abs() < 2e-3 + 0.05 * analytic.abs(),
                    "param {p} entry {idx}: numeric {numeric} vs analytic {analytic}"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, model.num_params());
    }

    #[test]
    fn threads_give_the_same_answer() {
        let (model, x, y) = tiny();
        let (l1, g1) = model.loss_and_grads(&x, &y, 4, 1);
        let (l2, g2) = model.loss_and_grads(&x, &y, 4, 2);
        assert!((l1 - l2).abs() < 1e-5);
        for (a, b) in g1.iter().zip(&g2) {
            for (u, v) in a.data.iter().zip(&b.data) {
                assert!((u - v).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn model_cannot_see_the_future() {
        let (model, x, _) = tiny();
        let a = model.logits(&x[..4]);
        let mut changed = x[..4].to_vec();
        changed[3] = 6; // rewrite only the last token
        let b = model.logits(&changed);
        let v = model.cfg.vocab_size;
        assert_eq!(&a.data[..3 * v], &b.data[..3 * v]); // earlier positions identical
        assert_ne!(&a.data[3 * v..], &b.data[3 * v..]);
    }

    #[test]
    fn it_can_memorise_a_sequence() {
        // The whole stack (embeddings, attention, LayerNorm, GELU, AdamW) has to
        // work together for this to succeed.
        let cfg = TransformerConfig { vocab_size: 5, ctx: 8, d_model: 16, n_layer: 1, n_head: 2 };
        let mut rng = Rng::new(5);
        let mut model = Transformer::new(cfg, &mut rng);
        let inputs = vec![0, 1, 2, 3, 4, 0, 1, 2];
        let targets = vec![1, 2, 3, 4, 0, 1, 2, 3];
        let mut opt = AdamW::new(&model.params, 1e-2, 0.0);
        let first = model.loss(&inputs, &targets, 8);
        for _ in 0..150 {
            let (_, grads) = model.loss_and_grads(&inputs, &targets, 8, 1);
            opt.step(&mut model.params, &grads);
        }
        let last = model.loss(&inputs, &targets, 8);
        assert!(last < 0.05 && last < first / 10.0, "loss {first} -> {last}");
    }
}
