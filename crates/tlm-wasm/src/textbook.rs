//! Bindings for the interactive textbook (`web/textbook/`).
//!
//! Each chapter's widgets call into the same engine the real model uses, so what the reader
//! plays with is not a mock-up. Chapter 1 needs only the tokenizer.

use crate::Session;
use serde::Serialize;
use tlm_core::bpe::Tokenizer;
use tlm_core::transformer::param_names;
use wasm_bindgen::prelude::*;

/// Training text is capped so building the lab stays quick on a phone.
const MAX_TRAIN_BYTES: usize = 1_000_000;
/// Most merges the lab will learn (vocabulary 256 + this).
const MAX_MERGES: usize = 2048;
/// Longest text a widget may encode in one call.
const MAX_ENCODE_BYTES: usize = 20_000;

#[derive(Serialize)]
struct TokenOut {
    id: usize,
    bytes: Vec<u8>,
}

#[derive(Serialize)]
struct MergeOut {
    id: usize,
    left: Vec<u8>,
    right: Vec<u8>,
}

/// A tokenizer trained once, whose vocabulary can then be shrunk to "the first k merges".
/// Because merges are learned in order, the first k merges are exactly the tokenizer you would
/// get by asking for a vocabulary of 256 + k.
#[wasm_bindgen]
pub struct TokenizerLab {
    full: Tokenizer,
    active: Tokenizer,
}

#[wasm_bindgen]
impl TokenizerLab {
    /// Learn up to `max_merges` merges from `text` (only its first megabyte is used).
    #[wasm_bindgen(constructor)]
    pub fn new(text: &str, max_merges: usize) -> Result<TokenizerLab, JsError> {
        if max_merges == 0 || max_merges > MAX_MERGES {
            return Err(JsError::new(&format!("max_merges must be between 1 and {MAX_MERGES}")));
        }
        let text = truncate_on_char(text, MAX_TRAIN_BYTES);
        let full = Tokenizer::train(text, 256 + max_merges);
        let active = with_merges(&full, full.merges.len());
        Ok(TokenizerLab { full, active })
    }

    /// How many merges were actually learned (fewer than asked if the text ran out of pairs).
    pub fn learned_merges(&self) -> usize {
        self.full.merges.len()
    }

    /// Use only the first `k` merges (vocabulary size 256 + k).
    pub fn set_merges(&mut self, k: usize) {
        self.active = with_merges(&self.full, k.min(self.full.merges.len()));
    }

    pub fn vocab_size(&self) -> usize {
        self.active.vocab_size()
    }

    /// Text -> JSON list of `{id, bytes}`, one entry per token, in order.
    pub fn encode(&self, text: &str) -> Result<String, JsError> {
        if text.len() > MAX_ENCODE_BYTES {
            return Err(JsError::new("text is too long for this widget"));
        }
        let out: Vec<TokenOut> = self
            .active
            .encode(text)
            .into_iter()
            .map(|id| TokenOut { id, bytes: self.active.vocab[id].clone() })
            .collect();
        Ok(serde_json::to_string(&out).expect("tokens are always serialisable"))
    }

    /// Token ids -> text. Ids outside the current vocabulary are an error, not a panic.
    pub fn decode(&self, ids: &[u32]) -> Result<String, JsError> {
        let mut checked = Vec::with_capacity(ids.len());
        for &id in ids {
            if id as usize >= self.active.vocab_size() {
                return Err(JsError::new(&format!("token id {id} is outside the vocabulary (size {})", self.active.vocab_size())));
            }
            checked.push(id as usize);
        }
        Ok(self.active.decode(&checked))
    }

    /// The first `count` merges (of the full list) as JSON `{id, left, right}`, with each side
    /// given as the bytes it stands for. This is the "recipe" the tokenizer learned.
    pub fn merges(&self, count: usize) -> String {
        let n = count.min(self.full.merges.len());
        let out: Vec<MergeOut> = self.full.merges[..n]
            .iter()
            .enumerate()
            .map(|(k, &(a, b))| MergeOut { id: 256 + k, left: self.full.vocab[a].clone(), right: self.full.vocab[b].clone() })
            .collect();
        serde_json::to_string(&out).expect("merges are always serialisable")
    }
}

/// A tokenizer made of only the first `k` merges of `full`.
fn with_merges(full: &Tokenizer, k: usize) -> Tokenizer {
    let merges: Vec<[usize; 2]> = full.merges[..k].iter().map(|&(a, b)| [a, b]).collect();
    let json = serde_json::json!({ "type": "byte-bpe", "merges": merges }).to_string();
    Tokenizer::from_json(&json).expect("a prefix of valid merges is valid")
}

/// Longest prefix of `s` that is at most `max` bytes and ends on a character boundary.
fn truncate_on_char(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

// ---- Chapter 3: attention -----------------------------------------------------

#[derive(Serialize)]
struct AttentionOut {
    tokens: Vec<TokenOut>,
    n_layer: usize,
    n_head: usize,
    /// True if the text had more tokens than the model's window and was cut.
    truncated: bool,
    /// `weights[layer][head]` = T x T numbers, row-major: row i = how token i spreads its attention.
    weights: Vec<Vec<Vec<f32>>>,
    /// Raw scores before the softmax, same layout. Entries right of the diagonal are unused.
    scores: Vec<Vec<Vec<f32>>>,
}

/// Run the session's current model on `text` and report what every attention head did.
/// Only the first `ctx` tokens are used (the model's window).
#[wasm_bindgen]
pub fn attention_for_text(session: &Session, text: &str) -> Result<String, JsError> {
    attention_json(session, text).map_err(js)
}

fn attention_json(session: &Session, text: &str) -> Result<String, String> {
    let model = session.model();
    let ctx = model.cfg.ctx;
    // Cut very long text first so encoding stays instant; the window is only `ctx` tokens anyway.
    let text = truncate_on_char(text, ctx * 16);
    let mut ids = session.tok.encode(text);
    if ids.is_empty() {
        return Err("type some text first".to_string());
    }
    let truncated = ids.len() > ctx;
    ids.truncate(ctx);
    let trace = model.attention_trace(&ids);
    let flat = |all: &Vec<Vec<tlm_core::Matrix>>| -> Vec<Vec<Vec<f32>>> { all.iter().map(|l| l.iter().map(|m| m.data.clone()).collect()).collect() };
    let out = AttentionOut {
        tokens: ids.iter().map(|&id| TokenOut { id, bytes: session.tok.vocab[id].clone() }).collect(),
        n_layer: model.cfg.n_layer,
        n_head: model.cfg.n_head,
        truncated,
        weights: flat(&trace.weights),
        scores: flat(&trace.scores),
    };
    let finite = |v: &Vec<Vec<Vec<f32>>>| v.iter().flatten().flatten().all(|x| x.is_finite());
    if !finite(&out.weights) || !finite(&out.scores) {
        return Err("the model produced a non-finite number (training may have diverged)".to_string());
    }
    Ok(serde_json::to_string(&out).expect("attention is always serialisable"))
}

// ---- Chapter 5: sampling ------------------------------------------------------

#[derive(Serialize)]
struct NextOut {
    /// The prompt's tokens (cut to the model's window, keeping the END of the text).
    tokens: Vec<TokenOut>,
    truncated: bool,
    /// The model's raw score for every token as the next one (the logits), index = token id.
    logits: Vec<f32>,
}

/// Every token's bytes, as a JSON list of byte lists (index = token id).
#[wasm_bindgen]
pub fn vocab_bytes(session: &Session) -> String {
    serde_json::to_string(&session.tok.vocab).expect("bytes are always serialisable")
}

/// The raw next-token scores after `text`, so the page can apply temperature itself and show every step.
#[wasm_bindgen]
pub fn next_logits(session: &Session, text: &str) -> Result<String, JsError> {
    next_logits_json(session, text).map_err(js)
}

fn next_logits_json(session: &Session, text: &str) -> Result<String, String> {
    let model = session.model();
    let ctx = model.cfg.ctx;
    let mut ids = session.tok.encode(text);
    if ids.is_empty() {
        return Err("type some text first".to_string());
    }
    let truncated = ids.len() > ctx;
    if truncated {
        ids.drain(..ids.len() - ctx);
    }
    let logits = model.logits(&ids);
    let v = logits.cols;
    let last = logits.data[(ids.len() - 1) * v..ids.len() * v].to_vec();
    if !last.iter().all(|x| x.is_finite()) {
        return Err("the model produced a non-finite number (training may have diverged)".to_string());
    }
    let out = NextOut { tokens: ids.iter().map(|&id| TokenOut { id, bytes: session.tok.vocab[id].clone() }).collect(), truncated, logits: last };
    Ok(serde_json::to_string(&out).expect("logits are always serialisable"))
}

// ---- Chapter 4: learning ------------------------------------------------------

#[derive(Serialize)]
struct GuessOut {
    id: usize,
    bytes: Vec<u8>,
    prob: f32,
}

#[derive(Serialize)]
struct PositionOut {
    /// Probability the model gave the token that really came next (absent for the last token).
    p_actual: Option<f32>,
    /// The model's `top_k` favourite next tokens after this position, most likely first.
    top: Vec<GuessOut>,
}

#[derive(Serialize)]
struct PredictOut {
    tokens: Vec<TokenOut>,
    truncated: bool,
    vocab: usize,
    /// Mean of -ln(p_actual) over the positions that have a next token (the training loss, on this text).
    mean_loss: f32,
    positions: Vec<PositionOut>,
}

/// For every position of `text`, how likely the session's current model thought the real next token
/// was, and what it would have guessed instead. Only the first `ctx` tokens are used.
#[wasm_bindgen]
pub fn predict_for_text(session: &Session, text: &str, top_k: usize) -> Result<String, JsError> {
    predict_json(session, text, top_k).map_err(js)
}

fn predict_json(session: &Session, text: &str, top_k: usize) -> Result<String, String> {
    let model = session.model();
    let ctx = model.cfg.ctx;
    let text = truncate_on_char(text, ctx * 16);
    let mut ids = session.tok.encode(text);
    if ids.len() < 2 {
        return Err("type a little more text first".to_string());
    }
    let truncated = ids.len() > ctx;
    ids.truncate(ctx);
    let logits = model.logits(&ids);
    let k = top_k.clamp(1, 20);
    let mut positions = Vec::with_capacity(ids.len());
    let (mut loss_sum, mut loss_n) = (0.0f32, 0usize);
    for i in 0..ids.len() {
        let probs = tlm_core::ops::softmax(&logits.data[i * logits.cols..(i + 1) * logits.cols]);
        let p_actual = ids.get(i + 1).map(|&next| probs[next]);
        if let Some(p) = p_actual {
            loss_sum += -p.max(1e-30).ln();
            loss_n += 1;
        }
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
        let top = order.into_iter().take(k).map(|id| GuessOut { id, bytes: session.tok.vocab[id].clone(), prob: probs[id] }).collect();
        positions.push(PositionOut { p_actual, top });
    }
    let mean_loss = loss_sum / loss_n as f32;
    if !mean_loss.is_finite() {
        return Err("the model produced a non-finite number (training may have diverged)".to_string());
    }
    let out = PredictOut {
        tokens: ids.iter().map(|&id| TokenOut { id, bytes: session.tok.vocab[id].clone() }).collect(),
        truncated,
        vocab: model.cfg.vocab_size,
        mean_loss,
        positions,
    };
    Ok(serde_json::to_string(&out).expect("predictions are always serialisable"))
}

// ---- Chapter 2: embeddings ----------------------------------------------------

/// Power-iteration rounds for the PCA directions. 200 is far more than a 512 x 32 table needs.
const PCA_ITERS: usize = 200;

#[derive(Serialize)]
struct NeighborOut {
    id: usize,
    similarity: f32,
}

/// A frozen copy of a model's token-embedding table (one row of `dim` numbers per token),
/// plus the tools the chapter needs to look at it: neighbours and a 2-D map.
///
/// "Frozen" matters: the reader trains a model and keeps a snapshot every few hundred steps,
/// then scrubs through them to watch the table change.
#[wasm_bindgen]
pub struct EmbeddingView {
    table: Vec<f32>,
    vocab: usize,
    dim: usize,
    token_bytes: Vec<Vec<u8>>,
    /// Column means and the two principal directions of THIS table (unit length each).
    mean: Vec<f32>,
    comps: [Vec<f32>; 2],
    /// Share of the table's total variance along each of the two directions (0..=1).
    explained: [f32; 2],
}

#[wasm_bindgen]
impl EmbeddingView {
    /// Copy the token-embedding table out of a session's current model.
    #[wasm_bindgen(constructor)]
    pub fn new(session: &Session) -> Result<EmbeddingView, JsError> {
        let model = session.model();
        // The table is parameter 0; check by name so a layout change fails loudly, not silently.
        if param_names(&model.cfg)[0] != "tok_emb" {
            return Err(JsError::new("parameter 0 is not the token embedding table"));
        }
        let m = &model.params[0];
        let token_bytes: Vec<Vec<u8>> = (0..m.rows).map(|id| session.tok.vocab[id].clone()).collect();
        Ok(EmbeddingView::from_table(m.data.clone(), m.rows, m.cols, token_bytes))
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The bytes token `id` stands for.
    pub fn token_bytes(&self, id: usize) -> Result<Vec<u8>, JsError> {
        self.check(id).map_err(js)?;
        Ok(self.token_bytes[id].clone())
    }

    /// Every token's bytes, as a JSON list of byte lists (index = token id).
    pub fn vocab_json(&self) -> String {
        serde_json::to_string(&self.token_bytes).expect("bytes are always serialisable")
    }

    /// The `dim` numbers that stand for token `id`.
    pub fn vector(&self, id: usize) -> Result<Vec<f32>, JsError> {
        self.check(id).map_err(js)?;
        Ok(self.row(id).to_vec())
    }

    /// The `k` tokens whose vectors point most nearly the same way as token `id`'s (cosine
    /// similarity, highest first), as JSON `{id, similarity}`. The token itself is excluded.
    pub fn neighbors(&self, id: usize, k: usize) -> Result<String, JsError> {
        self.check(id).map_err(js)?;
        let me = self.row(id);
        let mut all: Vec<NeighborOut> = (0..self.vocab)
            .filter(|&j| j != id)
            .map(|j| NeighborOut { id: j, similarity: cosine(me, self.row(j)) })
            .collect();
        all.sort_by(|a, b| b.similarity.total_cmp(&a.similarity).then(a.id.cmp(&b.id)));
        all.truncate(k.min(200));
        Ok(serde_json::to_string(&all).expect("neighbours are always serialisable"))
    }

    /// Share of the variation this table has along its own two main directions, `[first, second]`.
    pub fn explained_variance(&self) -> Vec<f32> {
        self.explained.to_vec()
    }

    /// Where every token lands on this table's own 2-D map: `[x0, y0, x1, y1, ...]`.
    pub fn map(&self) -> Vec<f32> {
        self.project(&self.mean, &self.comps)
    }

    /// Where every token of THIS table lands when using `basis`'s map axes. Using the final
    /// model's axes for every snapshot keeps the picture steady while the reader scrubs.
    pub fn map_with(&self, basis: &EmbeddingView) -> Result<Vec<f32>, JsError> {
        self.map_with_impl(basis).map_err(js)
    }
}

impl EmbeddingView {
    /// Build a view from a raw row-major table. Also used by the tests.
    fn from_table(table: Vec<f32>, vocab: usize, dim: usize, token_bytes: Vec<Vec<u8>>) -> EmbeddingView {
        assert_eq!(table.len(), vocab * dim);
        assert_eq!(token_bytes.len(), vocab);
        let (mean, comps, explained) = pca(&table, vocab, dim);
        EmbeddingView { table, vocab, dim, token_bytes, mean, comps, explained }
    }

    fn row(&self, id: usize) -> &[f32] {
        &self.table[id * self.dim..(id + 1) * self.dim]
    }

    fn check(&self, id: usize) -> Result<(), String> {
        if id < self.vocab {
            Ok(())
        } else {
            Err(format!("token id {id} is outside the vocabulary (size {})", self.vocab))
        }
    }

    fn map_with_impl(&self, basis: &EmbeddingView) -> Result<Vec<f32>, String> {
        if basis.dim != self.dim {
            return Err("the two tables have different sizes".to_string());
        }
        Ok(self.project(&basis.mean, &basis.comps))
    }

    fn project(&self, mean: &[f32], comps: &[Vec<f32>; 2]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.vocab * 2);
        for id in 0..self.vocab {
            let r = self.row(id);
            for c in comps {
                out.push(r.iter().zip(mean).zip(c).map(|((x, m), w)| (x - m) * w).sum());
            }
        }
        out
    }
}

/// Turn an internal error message into a JavaScript exception (only possible inside a browser).
fn js(msg: String) -> JsError {
    JsError::new(&msg)
}

/// Cosine similarity: 1 = same direction, 0 = unrelated, -1 = opposite. A zero vector has no
/// direction, so it is "unrelated" to everything (0) rather than NaN.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Top two principal directions of a `rows x dim` table by power iteration (no libraries).
/// Returns (column means, [dir1, dir2], [variance share 1, variance share 2]).
/// Signs are fixed so the same table always gives the same picture.
fn pca(table: &[f32], rows: usize, dim: usize) -> (Vec<f32>, [Vec<f32>; 2], [f32; 2]) {
    let mut mean = vec![0.0f32; dim];
    for r in 0..rows {
        for c in 0..dim {
            mean[c] += table[r * dim + c];
        }
    }
    mean.iter_mut().for_each(|m| *m /= rows.max(1) as f32);
    let centered: Vec<f32> = (0..rows * dim).map(|i| table[i] - mean[i % dim]).collect();
    let total: f32 = centered.iter().map(|x| x * x).sum();

    let mut dirs: Vec<Vec<f32>> = Vec::new();
    let mut shares = [0.0f32; 2];
    for k in 0..2 {
        // Fixed, non-degenerate starting vector.
        let mut v: Vec<f32> = (0..dim).map(|i| 1.0 + ((i * 7 + k * 3) % 5) as f32 * 0.37).collect();
        for _ in 0..PCA_ITERS {
            orthogonalize(&mut v, &dirs);
            if !normalize(&mut v) {
                break;
            }
            // w = Xt (X v): one round of power iteration on the covariance matrix.
            let xv: Vec<f32> = (0..rows).map(|r| centered[r * dim..(r + 1) * dim].iter().zip(&v).map(|(a, b)| a * b).sum()).collect();
            let mut w = vec![0.0f32; dim];
            for r in 0..rows {
                for c in 0..dim {
                    w[c] += centered[r * dim + c] * xv[r];
                }
            }
            v = w;
        }
        orthogonalize(&mut v, &dirs);
        if !normalize(&mut v) {
            v = vec![0.0; dim];
            if dim > k {
                v[k] = 1.0; // degenerate table (all rows equal): any axis will do
            }
        }
        // Fix the sign: the largest-magnitude entry is positive.
        let big = v.iter().cloned().fold(0.0f32, |a, b| if b.abs() > a.abs() { b } else { a });
        if big < 0.0 {
            v.iter_mut().for_each(|x| *x = -*x);
        }
        let var: f32 = (0..rows)
            .map(|r| {
                let p: f32 = centered[r * dim..(r + 1) * dim].iter().zip(&v).map(|(a, b)| a * b).sum();
                p * p
            })
            .sum();
        shares[k] = if total > 0.0 { var / total } else { 0.0 };
        dirs.push(v);
    }
    let d1 = dirs.pop().unwrap();
    let d0 = dirs.pop().unwrap();
    (mean, [d0, d1], shares)
}

/// Remove from `v` its component along each (unit-length) direction in `dirs`.
fn orthogonalize(v: &mut [f32], dirs: &[Vec<f32>]) {
    for d in dirs {
        let dot: f32 = v.iter().zip(d).map(|(a, b)| a * b).sum();
        v.iter_mut().zip(d).for_each(|(a, b)| *a -= dot * b);
    }
}

/// Scale `v` to length 1. Returns false if it has no length to scale.
fn normalize(v: &mut [f32]) -> bool {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n < 1e-20 || !n.is_finite() {
        return false;
    }
    v.iter_mut().for_each(|x| *x /= n);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- attention ----

    fn small_session(steps: usize) -> Session {
        let text = "Once upon a time a little girl named Lily found a shiny red ball in the garden. The dog and the cat saw the ball too.\n<|endoftext|>\n".repeat(120);
        let cfg = format!(r#"{{"vocab_size": 300, "d_model": 16, "n_layer": 2, "n_head": 2, "ctx": 32, "batch": 4, "steps": 100, "max_bytes": 20000}}"#);
        let mut s = Session::new(&text, &cfg).unwrap();
        s.step(steps);
        s
    }

    #[test]
    fn next_logits_match_the_models_own_last_row() {
        let session = small_session(5);
        let text = "The dog and the cat saw the";
        let v: serde_json::Value = serde_json::from_str(&next_logits_json(&session, text).unwrap()).unwrap();
        let got: Vec<f32> = v["logits"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let ids = session.tok.encode(text);
        let all = session.model().logits(&ids);
        let want = &all.data[(ids.len() - 1) * all.cols..];
        assert_eq!(got.len(), all.cols);
        assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-4));
        assert!(next_logits_json(&session, "").is_err());
        let vocab: Vec<Vec<u8>> = serde_json::from_str(&vocab_bytes(&session)).unwrap();
        assert_eq!(vocab.len(), all.cols);
    }

    #[test]
    fn predict_json_probabilities_and_loss_are_consistent() {
        let session = small_session(5);
        let text = "The dog and the cat saw the ball.";
        let v: serde_json::Value = serde_json::from_str(&predict_json(&session, text, 5).unwrap()).unwrap();
        let n = v["tokens"].as_array().unwrap().len();
        let pos = v["positions"].as_array().unwrap();
        assert_eq!(pos.len(), n);
        assert!(pos[n - 1]["p_actual"].is_null() && !pos[0]["p_actual"].is_null());
        let mut sum = 0.0;
        for p in &pos[..n - 1] {
            sum += -p["p_actual"].as_f64().unwrap().ln();
            let top = p["top"].as_array().unwrap();
            assert_eq!(top.len(), 5);
            assert!(top.windows(2).all(|w| w[0]["prob"].as_f64() >= w[1]["prob"].as_f64()));
        }
        let mean = v["mean_loss"].as_f64().unwrap();
        assert!((mean - sum / (n - 1) as f64).abs() < 1e-4);
        // An untrained model is near uniform: loss close to ln(vocab).
        let fresh = small_session(0);
        let f: serde_json::Value = serde_json::from_str(&predict_json(&fresh, text, 3).unwrap()).unwrap();
        let ln_v = (f["vocab"].as_f64().unwrap()).ln();
        assert!((f["mean_loss"].as_f64().unwrap() - ln_v).abs() < 1.0, "{} vs {}", f["mean_loss"], ln_v);
        assert!(predict_json(&session, "a", 3).is_err());
    }

    #[test]
    fn attention_json_has_one_causal_distribution_per_row() {
        let session = small_session(5);
        let text = "The dog and the cat saw the ball.";
        let v: serde_json::Value = serde_json::from_str(&attention_json(&session, text).unwrap()).unwrap();
        let toks = v["tokens"].as_array().unwrap();
        let t = toks.len();
        assert!(t > 3 && !v["truncated"].as_bool().unwrap());
        let bytes: Vec<u8> = toks.iter().flat_map(|x| x["bytes"].as_array().unwrap().iter().map(|b| b.as_u64().unwrap() as u8)).collect();
        assert_eq!(String::from_utf8(bytes).unwrap(), text);
        assert_eq!((v["n_layer"].as_u64(), v["n_head"].as_u64()), (Some(2), Some(2)));
        for l in 0..2 {
            for h in 0..2 {
                let w: Vec<f64> = v["weights"][l][h].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
                let s = v["scores"][l][h].as_array().unwrap();
                assert_eq!((w.len(), s.len()), (t * t, t * t));
                for i in 0..t {
                    let sum: f64 = (0..=i).map(|j| w[i * t + j]).sum();
                    assert!((sum - 1.0).abs() < 1e-4);
                    assert!((i + 1..t).all(|j| w[i * t + j] == 0.0));
                }
            }
        }
    }

    #[test]
    fn attention_json_cuts_to_the_window_and_rejects_empty_text() {
        let session = small_session(1);
        let long = "the dog and the cat saw the ball ".repeat(40);
        let v: serde_json::Value = serde_json::from_str(&attention_json(&session, &long).unwrap()).unwrap();
        assert_eq!(v["tokens"].as_array().unwrap().len(), 32);
        assert!(v["truncated"].as_bool().unwrap());
        assert!(attention_json(&session, "").is_err());
        // text made only of multi-byte characters is cut on a character boundary, not mid-way
        assert!(attention_json(&session, &"日本語".repeat(300)).is_ok());
    }

    #[test]
    fn training_changes_what_attention_does() {
        let fresh = small_session(0);
        let trained = small_session(80);
        let text = "The dog and the cat saw the ball.";
        let a = attention_json(&fresh, text).unwrap();
        let b = attention_json(&trained, text).unwrap();
        assert_ne!(a, b);
    }

    // ---- embeddings ----

    fn toy_view(rows: &[&[f32]]) -> EmbeddingView {
        let dim = rows[0].len();
        let table: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let bytes = (0..rows.len()).map(|i| vec![i as u8]).collect();
        EmbeddingView::from_table(table, rows.len(), dim, bytes)
    }

    #[test]
    fn neighbors_rank_by_direction_and_exclude_self() {
        // 0 and 1 point almost the same way (1 is 3x longer: cosine ignores length); 2 is orthogonal; 3 opposite.
        let v = toy_view(&[&[1.0, 0.1, 0.0], &[3.0, 0.2, 0.0], &[0.0, 0.0, 1.0], &[-1.0, -0.1, 0.0], &[0.0, 0.0, 0.0]]);
        let n: serde_json::Value = serde_json::from_str(&v.neighbors(0, 10).unwrap()).unwrap();
        let ids: Vec<u64> = n.as_array().unwrap().iter().map(|e| e["id"].as_u64().unwrap()).collect();
        assert_eq!(ids[0], 1);
        assert!(!ids.contains(&0), "a token is not its own neighbour");
        assert_eq!(*ids.last().unwrap(), 3, "the opposite vector ranks last");
        assert!(n[0]["similarity"].as_f64().unwrap() > 0.99);
        // the zero vector is "unrelated" (0), never NaN
        let z: serde_json::Value = serde_json::from_str(&v.neighbors(4, 3).unwrap()).unwrap();
        assert!(z.as_array().unwrap().iter().all(|e| e["similarity"].as_f64().unwrap() == 0.0));
        assert_eq!(cosine(&[1.0, 2.0], &[2.0, 4.0]), 1.0);
    }

    #[test]
    fn pca_finds_the_line_and_is_deterministic() {
        // Points on the line y = 2x in 3-D, plus a tiny wobble in z: one direction explains nearly everything.
        let rows: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                let x = i as f32 - 9.5;
                vec![x, 2.0 * x, 0.01 * ((i * 5 % 7) as f32 - 3.0)]
            })
            .collect();
        let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
        let v = toy_view(&refs);
        assert!(v.explained[0] > 0.999, "first direction explains {}", v.explained[0]);
        assert!(v.explained[0] + v.explained[1] <= 1.0001);
        let d = &v.comps[0];
        assert!((d[1] / d[0] - 2.0).abs() < 0.01, "direction should have slope 2, got {d:?}");
        assert!(d[0] > 0.0, "sign is fixed");
        let again = toy_view(&refs);
        assert_eq!(v.map(), again.map());
        // the second direction is perpendicular to the first
        let dot: f32 = v.comps[0].iter().zip(&v.comps[1]).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-3);
    }

    #[test]
    fn pca_survives_degenerate_tables() {
        let same = toy_view(&[&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]]);
        assert!(same.map().iter().all(|x| x.is_finite() && x.abs() < 1e-5));
        assert_eq!(same.explained, [0.0, 0.0]);
        let tiny = toy_view(&[&[5.0]]);
        assert!(tiny.map().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn map_with_uses_the_given_axes() {
        let a = toy_view(&[&[0.0, 0.0], &[2.0, 0.0], &[4.0, 0.0]]);
        let b = toy_view(&[&[0.0, 0.0], &[1.0, 0.0], &[2.0, 0.0]]);
        let on_a = b.map_with_impl(&a).unwrap();
        // a's first axis is the x direction, so b's points keep their order and spacing along it
        assert!(on_a[0] < on_a[2] && on_a[2] < on_a[4]);
        assert!(((on_a[2] - on_a[0]) - 1.0).abs() < 1e-4);
        let wrong = toy_view(&[&[0.0, 0.0, 0.0]]);
        assert!(a.map_with_impl(&wrong).is_err());
    }

    #[test]
    fn out_of_range_ids_are_errors() {
        let v = toy_view(&[&[1.0, 0.0], &[0.0, 1.0]]);
        assert!(v.check(2).is_err());
        assert!(v.check(1).is_ok());
        assert!(v.check(usize::MAX).is_err());
    }

    #[test]
    fn snapshot_of_a_real_session_changes_with_training() {
        let text = "Once upon a time a little girl named Lily found a shiny red ball in the garden. The dog and the cat saw the ball too.\n<|endoftext|>\n".repeat(120);
        let cfg = r#"{"vocab_size": 300, "d_model": 16, "n_layer": 1, "n_head": 2, "ctx": 32, "batch": 4, "steps": 60, "max_bytes": 20000}"#;
        let mut session = Session::new(&text, cfg).unwrap();
        let before = EmbeddingView::new(&session).unwrap();
        assert_eq!((before.vocab_size(), before.dim()), (session.vocab_size(), 16));
        session.step(40);
        let after = EmbeddingView::new(&session).unwrap();
        assert_ne!(before.table, after.table, "training must move the embeddings");
        assert_eq!(before.token_bytes, after.token_bytes);
        assert_eq!(after.map().len(), 2 * after.vocab_size());
        assert!(after.map_with_impl(&before).unwrap().iter().all(|x| x.is_finite()));
        assert_eq!(after.token_bytes(65).unwrap(), b"A".to_vec());
    }

    fn sample() -> String {
        "the cat sat on the mat. the dog sat on the log. ".repeat(200)
    }

    #[test]
    fn prefix_of_merges_equals_smaller_vocabulary() {
        let text = sample();
        let big = TokenizerLab::new(&text, 40).unwrap();
        let small = Tokenizer::train(&text, 256 + 10);
        let mut lab = big;
        lab.set_merges(10);
        assert_eq!(lab.vocab_size(), small.vocab_size());
        assert_eq!(lab.active.encode("the cat sat"), small.encode("the cat sat"));
    }

    #[test]
    fn more_merges_means_fewer_tokens_and_round_trip_holds() {
        let text = sample();
        let mut lab = TokenizerLab::new(&text, 60).unwrap();
        lab.set_merges(0);
        let bytes_only = lab.active.encode("the cat sat on the mat").len();
        assert_eq!(bytes_only, "the cat sat on the mat".len());
        lab.set_merges(60);
        let merged = lab.active.encode("the cat sat on the mat");
        assert!(merged.len() < bytes_only);
        assert_eq!(lab.active.decode(&merged), "the cat sat on the mat");
    }

    #[test]
    fn unicode_and_empty_text_are_handled() {
        let lab = TokenizerLab::new(&sample(), 20).unwrap();
        let json = lab.encode("héllo 🌍").unwrap();
        let toks: serde_json::Value = serde_json::from_str(&json).unwrap();
        let bytes: Vec<u8> = toks.as_array().unwrap().iter().flat_map(|t| t["bytes"].as_array().unwrap().iter().map(|b| b.as_u64().unwrap() as u8)).collect();
        assert_eq!(String::from_utf8(bytes).unwrap(), "héllo 🌍");
        assert_eq!(lab.encode("").unwrap(), "[]");
    }

    #[test]
    fn set_merges_beyond_learned_is_clamped_and_truncation_is_char_safe() {
        let mut lab = TokenizerLab::new(&sample(), 20).unwrap();
        lab.set_merges(10_000);
        assert_eq!(lab.vocab_size(), 256 + lab.learned_merges());
        assert_eq!(truncate_on_char("aé", 2), "a"); // é is 2 bytes; cutting at 2 would split it
        assert_eq!(truncate_on_char("abc", 10), "abc");
    }

    #[test]
    fn merge_list_describes_the_first_merges() {
        let lab = TokenizerLab::new(&sample(), 5).unwrap();
        let m: serde_json::Value = serde_json::from_str(&lab.merges(3)).unwrap();
        assert_eq!(m.as_array().unwrap().len(), 3);
        assert_eq!(m[0]["id"], 256);
    }
}
