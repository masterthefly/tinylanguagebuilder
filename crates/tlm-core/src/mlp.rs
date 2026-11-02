//! Rung 2 of the model ladder: look at the last few characters, not just one.
//!
//!   context ids -> embedding lookup -> concat -> Linear+tanh -> Linear -> logits
//!
//! Forward AND backward passes are written by hand (no autograd), and the
//! tests verify every gradient numerically.

use crate::ops::{sample_from_probs, softmax, softmax_rows};
use crate::{Matrix, Rng};

// Indices into `Mlp::params` (and into the gradient list it returns).
pub const EMB: usize = 0; // vocab_size x emb_dim
pub const W1: usize = 1; // (context * emb_dim) x hidden
pub const B1: usize = 2; // 1 x hidden
pub const W2: usize = 3; // hidden x vocab_size
pub const B2: usize = 4; // 1 x vocab_size

#[derive(Debug, Clone, Copy)]
pub struct MlpConfig {
    pub vocab_size: usize,
    pub context: usize,
    pub emb_dim: usize,
    pub hidden: usize,
}

pub struct Mlp {
    pub cfg: MlpConfig,
    pub params: Vec<Matrix>,
}

/// Intermediate values from the forward pass, needed again by backward.
struct Cache {
    x: Matrix,      // N x (context*emb_dim): the concatenated embeddings
    h: Matrix,      // N x hidden:            after tanh
    logits: Matrix, // N x vocab_size
}

/// Random numbers uniformly in [-1/sqrt(fan_in), +1/sqrt(fan_in)].
/// Small weights keep early activations in tanh's useful range.
pub(crate) fn random_matrix(rows: usize, cols: usize, fan_in: usize, rng: &mut Rng) -> Matrix {
    let scale = 1.0 / (fan_in as f32).sqrt();
    let data = (0..rows * cols).map(|_| (rng.next_f32() * 2.0 - 1.0) * scale).collect();
    Matrix::from_vec(rows, cols, data)
}

/// Add a 1 x cols bias row to every row of `m`.
pub(crate) fn add_bias(m: &mut Matrix, bias: &Matrix) {
    for r in 0..m.rows {
        for c in 0..m.cols {
            m.data[r * m.cols + c] += bias.data[c];
        }
    }
}

/// Collapse N x cols into 1 x cols by summing down each column.
pub(crate) fn sum_rows(m: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(1, m.cols);
    for r in 0..m.rows {
        for c in 0..m.cols {
            out.data[c] += m.data[r * m.cols + c];
        }
    }
    out
}

impl Mlp {
    pub fn new(cfg: MlpConfig, rng: &mut Rng) -> Self {
        let MlpConfig { vocab_size: v, context: c, emb_dim: d, hidden: h } = cfg;
        let params = vec![
            random_matrix(v, d, d, rng),         // EMB
            random_matrix(c * d, h, c * d, rng), // W1
            Matrix::zeros(1, h),                 // B1
            random_matrix(h, v, h, rng),         // W2
            Matrix::zeros(1, v),                 // B2
        ];
        Mlp { cfg, params }
    }

    pub fn num_params(&self) -> usize {
        self.params.iter().map(|p| p.data.len()).sum()
    }

    /// `ctx` holds N examples back to back, each `context` ids long.
    fn forward(&self, ctx: &[usize]) -> Cache {
        let MlpConfig { context: c, emb_dim: d, .. } = self.cfg;
        assert_eq!(ctx.len() % c, 0, "ctx length must be a multiple of context");
        let n = ctx.len() / c;

        // 1. Embedding lookup: copy each id's row into place.
        let emb = &self.params[EMB];
        let mut x = Matrix::zeros(n, c * d);
        for i in 0..n {
            for j in 0..c {
                let id = ctx[i * c + j];
                let dst = i * (c * d) + j * d;
                x.data[dst..dst + d].copy_from_slice(&emb.data[id * d..(id + 1) * d]);
            }
        }

        // 2. Layer 1 + tanh.
        let mut h = x.matmul(&self.params[W1]);
        add_bias(&mut h, &self.params[B1]);
        for v in h.data.iter_mut() {
            *v = v.tanh();
        }

        // 3. Layer 2 gives raw scores.
        let mut logits = h.matmul(&self.params[W2]);
        add_bias(&mut logits, &self.params[B2]);

        Cache { x, h, logits }
    }

    /// Average cross-entropy loss only (no gradients). Use for validation.
    pub fn loss(&self, ctx: &[usize], targets: &[usize]) -> f32 {
        let cache = self.forward(ctx);
        let probs = softmax_rows(&cache.logits);
        let total: f32 = targets.iter().enumerate().map(|(i, &t)| -probs.get(i, t).ln()).sum();
        total / targets.len() as f32
    }

    /// Loss plus the gradient for every parameter (same order as `params`).
    pub fn loss_and_grads(&self, ctx: &[usize], targets: &[usize]) -> (f32, Vec<Matrix>) {
        let MlpConfig { vocab_size: v, context: c, emb_dim: d, .. } = self.cfg;
        let n = targets.len();
        let cache = self.forward(ctx);

        // Loss, and the gradient at the scores: (probs - one_hot) / N.
        let probs = softmax_rows(&cache.logits);
        let mut loss = 0.0;
        let mut dlogits = probs.clone();
        for (i, &t) in targets.iter().enumerate() {
            loss += -probs.get(i, t).ln();
            dlogits.data[i * v + t] -= 1.0;
        }
        for g in dlogits.data.iter_mut() {
            *g /= n as f32;
        }

        // Layer 2.
        let dw2 = cache.h.transpose().matmul(&dlogits);
        let db2 = sum_rows(&dlogits);
        let dh = dlogits.matmul(&self.params[W2].transpose());

        // Through tanh: derivative is 1 - tanh(z)^2, and h already IS tanh(z).
        let mut dpre = dh;
        for (g, &hv) in dpre.data.iter_mut().zip(&cache.h.data) {
            *g *= 1.0 - hv * hv;
        }

        // Layer 1.
        let dw1 = cache.x.transpose().matmul(&dpre);
        let db1 = sum_rows(&dpre);
        let dx = dpre.matmul(&self.params[W1].transpose());

        // Embeddings: route each slice of dx back to the row it was copied from.
        // (+= because the same character can appear many times.)
        let mut demb = Matrix::zeros(v, d);
        for i in 0..n {
            for j in 0..c {
                let id = ctx[i * c + j];
                for k in 0..d {
                    demb.data[id * d + k] += dx.data[i * (c * d) + j * d + k];
                }
            }
        }

        (loss / n as f32, vec![demb, dw1, db1, dw2, db2])
    }

    /// One gradient-descent step. Returns the loss before the update.
    pub fn train_step(&mut self, ctx: &[usize], targets: &[usize], lr: f32) -> f32 {
        let (loss, grads) = self.loss_and_grads(ctx, targets);
        for (p, g) in self.params.iter_mut().zip(&grads) {
            for (w, dw) in p.data.iter_mut().zip(&g.data) {
                *w -= lr * dw;
            }
        }
        loss
    }

    /// Generate `len` ids. `start` must contain at least `context` ids.
    pub fn sample(&self, start: &[usize], len: usize, rng: &mut Rng) -> Vec<usize> {
        let c = self.cfg.context;
        assert!(start.len() >= c);
        let mut history = start.to_vec();
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            let window = &history[history.len() - c..];
            let cache = self.forward(window);
            let probs = softmax(&cache.logits.data);
            let next = sample_from_probs(&probs, rng);
            history.push(next);
            out.push(next);
        }
        out
    }
}

/// Turn a token stream into (context, target) training pairs.
/// Returns (flat contexts, targets): pair i uses ids[i..i+context] to predict ids[i+context].
pub fn make_dataset(ids: &[usize], context: usize) -> (Vec<usize>, Vec<usize>) {
    let mut ctx = Vec::new();
    let mut targets = Vec::new();
    for i in 0..ids.len().saturating_sub(context) {
        ctx.extend_from_slice(&ids[i..i + context]);
        targets.push(ids[i + context]);
    }
    (ctx, targets)
}

/// Pick `batch` random examples from a dataset made by `make_dataset`.
/// Training on a random handful per step ("minibatch SGD") is far cheaper
/// than using every example, and the gradient is still a good estimate.
pub fn sample_batch(
    ctx: &[usize],
    targets: &[usize],
    context: usize,
    batch: usize,
    rng: &mut Rng,
) -> (Vec<usize>, Vec<usize>) {
    let mut bx = Vec::with_capacity(batch * context);
    let mut by = Vec::with_capacity(batch);
    for _ in 0..batch {
        let i = rng.next_usize(targets.len());
        bx.extend_from_slice(&ctx[i * context..(i + 1) * context]);
        by.push(targets[i]);
    }
    (bx, by)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_rows_stay_aligned() {
        // Each example is [i, i+1] -> target i+2, so alignment is checkable.
        let (ctx, targets) = make_dataset(&(0..100).collect::<Vec<_>>(), 2);
        let mut rng = Rng::new(5);
        let (bx, by) = sample_batch(&ctx, &targets, 2, 20, &mut rng);
        for i in 0..20 {
            assert_eq!(bx[i * 2 + 1], bx[i * 2] + 1);
            assert_eq!(by[i], bx[i * 2] + 2);
        }
    }

    fn small() -> (Mlp, Vec<usize>, Vec<usize>) {
        let cfg = MlpConfig { vocab_size: 5, context: 2, emb_dim: 3, hidden: 4 };
        let mut rng = Rng::new(3);
        let mlp = Mlp::new(cfg, &mut rng);
        // 4 examples, 2 context ids each.
        let ctx = vec![0, 1, 1, 2, 2, 3, 3, 4];
        let targets = vec![2, 3, 4, 0];
        (mlp, ctx, targets)
    }

    #[test]
    fn dataset_windows() {
        let (ctx, targets) = make_dataset(&[0, 1, 2, 3, 4], 2);
        assert_eq!(ctx, vec![0, 1, 1, 2, 2, 3]);
        assert_eq!(targets, vec![2, 3, 4]);
    }

    #[test]
    fn untrained_loss_is_near_ln_vocab() {
        let (mlp, ctx, targets) = small();
        let loss = mlp.loss(&ctx, &targets);
        assert!((loss - 5.0f32.ln()).abs() < 0.3, "loss was {loss}");
    }

    #[test]
    fn all_gradients_match_numerical_estimate() {
        let (mut mlp, ctx, targets) = small();
        let (_, grads) = mlp.loss_and_grads(&ctx, &targets);
        let h = 1e-2;
        for p in 0..mlp.params.len() {
            for idx in 0..mlp.params[p].data.len() {
                let original = mlp.params[p].data[idx];
                mlp.params[p].data[idx] = original + h;
                let up = mlp.loss(&ctx, &targets);
                mlp.params[p].data[idx] = original - h;
                let down = mlp.loss(&ctx, &targets);
                mlp.params[p].data[idx] = original;
                let numeric = (up - down) / (2.0 * h);
                let analytic = grads[p].data[idx];
                assert!(
                    (numeric - analytic).abs() < 2e-3,
                    "param {p} entry {idx}: numeric {numeric} vs analytic {analytic}"
                );
            }
        }
    }

    #[test]
    fn training_lowers_loss() {
        let (mut mlp, ctx, targets) = small();
        let first = mlp.train_step(&ctx, &targets, 0.5);
        let mut last = first;
        for _ in 0..300 {
            last = mlp.train_step(&ctx, &targets, 0.5);
        }
        assert!(last < first * 0.2, "loss should fall a lot: {first} -> {last}");
    }
}
