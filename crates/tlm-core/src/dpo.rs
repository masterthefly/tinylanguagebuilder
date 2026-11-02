//! Direct Preference Optimization (DPO): learn from pairs of (better, worse) answers.
//!
//! For one prompt with a `chosen` and a `rejected` answer, let `lp(y)` be the sum of
//! log-probabilities the model gives the answer's tokens. DPO compares the model with a
//! frozen REFERENCE copy of itself (the model before DPO started):
//!
//! ```text
//! z    = beta * [ (lp(chosen) - lp(rejected))  -  (lp_ref(chosen) - lp_ref(rejected)) ]
//! loss = -log sigmoid(z)
//! ```
//!
//! `z` grows when the model favours the chosen answer MORE than the reference did, so
//! minimising the loss moves probability from rejected answers to chosen ones, while the
//! reference term stops the model drifting arbitrarily far (`beta` sets how strongly).
//!
//! The gradient needs no new machinery. With `w = beta * sigmoid(-z)` it equals
//!   `+w * grad NLL(chosen)  -  w * grad NLL(rejected)`
//! which is the ordinary cross-entropy gradient with per-sequence weights, one of them
//! negative (`Transformer::weighted_grads`).
//!
//! Pairs come from the model itself ("on-policy") and a VERIFIER that can tell good answers
//! from bad ones: sample several answers per prompt, and pair a good one with a bad one.

use crate::bpe::Tokenizer;
use crate::optim::{clip_grad_norm, AdamW};
use crate::parallel::shard_range;
use crate::sft::{pack, TokenizedExample, STOP};
use crate::trainer::lr_at;
use crate::transformer::Transformer;
use crate::{Matrix, Rng};

// ---- small helpers -----------------------------------------------------------------

/// The word asked for in a "Word: X\nSentence:" prompt.
pub fn word_of(prompt: &str) -> &str {
    prompt.trim_start_matches("Word: ").split('\n').next().unwrap_or("")
}

/// -log(sigmoid(z)) = log(1 + e^-z), computed without overflow for large |z|.
fn softplus_neg(z: f32) -> f32 {
    if z > 0.0 { (-z).exp().ln_1p() } else { -z + z.exp().ln_1p() }
}

/// sigmoid(-z) = 1 / (1 + e^z), computed without overflow for large |z|.
fn sigmoid_neg(z: f32) -> f32 {
    if z >= 0.0 {
        let e = (-z).exp();
        e / (1.0 + e)
    } else {
        1.0 / (1.0 + z.exp())
    }
}

/// Apply `f` to every item, split across threads. Results come back IN ORDER, so the
/// answer does not depend on the thread count.
fn par_map<T: Sync, R: Send>(items: &[T], threads: usize, f: &(dyn Fn(usize, &T) -> R + Sync)) -> Vec<R> {
    let n = items.len();
    let threads = if cfg!(target_arch = "wasm32") { 1 } else { threads.clamp(1, n.max(1)) };
    if threads == 1 || n < 2 {
        return items.iter().enumerate().map(|(i, t)| f(i, t)).collect();
    }
    let chunk = n.div_ceil(threads);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|c| {
                let (lo, hi) = ((c * chunk).min(n), ((c + 1) * chunk).min(n));
                scope.spawn(move || (lo..hi).map(|i| f(i, &items[i])).collect::<Vec<R>>())
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("worker thread panicked")).collect()
    })
}

// ---- mining preference pairs ---------------------------------------------------------

/// One answer sampled from the model.
#[derive(Debug, Clone)]
pub struct Sample {
    /// The generated token ids, exactly as the model produced them.
    pub ids: Vec<usize>,
    pub text: String,
    /// Did the answer end with the stop marker (rather than hit the length limit)?
    pub stopped: bool,
}

/// Decides whether an answer is good: (answer text, stopped, the requested word) -> good?
pub type Verifier<'a> = &'a (dyn Fn(&str, bool, &str) -> bool + Sync);

/// The built-in verifier for the "use the word" task: the sentence must contain the
/// requested word AND finish properly.
pub fn uses_word_and_stops(text: &str, stopped: bool, word: &str) -> bool {
    stopped && crate::sft::contains_word(text, word)
}

pub fn sample_response(model: &Transformer, tok: &Tokenizer, prompt_ids: &[usize], max_new: usize, temperature: f32, rng: &mut Rng) -> Sample {
    let ids = model.sample_until(prompt_ids, max_new, temperature, rng, |g| tok.decode(g).contains(STOP));
    let full = tok.decode(&ids);
    match full.find(STOP) {
        Some(at) => Sample { ids, text: full[..at].to_string(), stopped: true },
        None => Sample { ids, text: full, stopped: false },
    }
}

/// From per-sample verdicts, pick one good and one bad answer (the first of each).
/// None if the samples are all good or all bad: there is nothing to compare.
pub fn pick_pair(good: &[bool]) -> Option<(usize, usize)> {
    Some((good.iter().position(|&g| g)?, good.iter().position(|&g| !g)?))
}

/// A mined pair, before the reference model has scored it.
#[derive(Debug, Clone)]
pub struct RawPair {
    pub prompt_ids: Vec<usize>,
    pub chosen: Vec<usize>,
    pub rejected: Vec<usize>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MineStats {
    pub prompts: usize,
    pub samples: usize,
    /// How many of the sampled answers the verifier liked.
    pub good: usize,
    pub pairs: usize,
}

/// Sample `k` answers for every prompt and pair a good one with a bad one when the
/// prompt produced both. Deterministic for a given `seed`, whatever the thread count.
///
/// `first_index` is the position of `prompts[0]` in the full prompt list: each prompt's
/// sampling is seeded by its GLOBAL index, so mining the list in several pieces (one per
/// browser worker) and joining the results in order gives exactly the same pairs as
/// mining it all at once.
#[allow(clippy::too_many_arguments)]
pub fn mine_pairs(
    model: &Transformer,
    tok: &Tokenizer,
    prompts: &[String],
    first_index: usize,
    k: usize,
    temperature: f32,
    max_new: usize,
    seed: u64,
    threads: usize,
    verify: Verifier,
) -> (Vec<RawPair>, MineStats) {
    let ctx = model.cfg.ctx;
    let per_prompt = par_map(prompts, threads, &|i, prompt: &String| {
        let prompt_ids = tok.encode(prompt);
        if prompt_ids.is_empty() || prompt_ids.len() + 2 > ctx {
            return (None, 0, 0);
        }
        // Prompt + answer must fit in the context window.
        let budget = max_new.min(ctx - 1 - prompt_ids.len());
        let mut rng = Rng::new(seed.wrapping_add(((first_index + i) as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
        let samples: Vec<Sample> = (0..k).map(|_| sample_response(model, tok, &prompt_ids, budget, temperature, &mut rng)).collect();
        let word = word_of(prompt);
        let good: Vec<bool> = samples.iter().map(|s| !s.ids.is_empty() && verify(&s.text, s.stopped, word)).collect();
        let n_good = good.iter().filter(|&&g| g).count();
        let pair = pick_pair(&good).map(|(c, r)| RawPair { prompt_ids: prompt_ids.clone(), chosen: samples[c].ids.clone(), rejected: samples[r].ids.clone() });
        (pair, samples.len(), n_good)
    });
    let mut stats = MineStats { prompts: prompts.len(), ..Default::default() };
    let mut pairs = Vec::new();
    for (pair, n, good) in per_prompt {
        stats.samples += n;
        stats.good += good;
        if let Some(p) = pair {
            pairs.push(p);
        }
    }
    stats.pairs = pairs.len();
    (pairs, stats)
}

// ---- moving pairs between workers and into storage -----------------------------------------

/// Little-endian reader that never trusts a length it has not checked.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn u32(&mut self) -> Result<u32, String> {
        let chunk = self.bytes.get(self.at..self.at + 4).ok_or("pair data ends unexpectedly")?;
        self.at += 4;
        Ok(u32::from_le_bytes(chunk.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn ids(&mut self, n: usize) -> Result<Vec<usize>, String> {
        // Check the whole run fits BEFORE allocating, so a corrupt length cannot exhaust memory.
        if n.checked_mul(4).is_none_or(|bytes| self.at + bytes > self.bytes.len()) {
            return Err("pair data has a length that runs past its end".into());
        }
        (0..n).map(|_| self.u32().map(|v| v as usize)).collect()
    }
}

fn put(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_ids(out: &mut Vec<u8>, ids: &[usize]) {
    ids.iter().for_each(|&i| put(out, i as u32));
}

/// Mining results as bytes: 4 numbers of statistics, then each pair as
/// (prompt length, chosen length, rejected length, the three id lists).
pub fn encode_mined(pairs: &[RawPair], stats: &MineStats) -> Vec<u8> {
    let mut out = Vec::new();
    for v in [stats.prompts, stats.samples, stats.good, pairs.len()] {
        put(&mut out, v as u32);
    }
    for p in pairs {
        for len in [p.prompt_ids.len(), p.chosen.len(), p.rejected.len()] {
            put(&mut out, len as u32);
        }
        put_ids(&mut out, &p.prompt_ids);
        put_ids(&mut out, &p.chosen);
        put_ids(&mut out, &p.rejected);
    }
    out
}

pub fn decode_mined(bytes: &[u8]) -> Result<(Vec<RawPair>, MineStats), String> {
    let mut r = Reader { bytes, at: 0 };
    let stats = MineStats { prompts: r.u32()? as usize, samples: r.u32()? as usize, good: r.u32()? as usize, pairs: r.u32()? as usize };
    let mut pairs = Vec::new();
    for _ in 0..stats.pairs {
        let (np, nc, nr) = (r.u32()? as usize, r.u32()? as usize, r.u32()? as usize);
        pairs.push(RawPair { prompt_ids: r.ids(np)?, chosen: r.ids(nc)?, rejected: r.ids(nr)? });
    }
    if r.at != bytes.len() {
        return Err("pair data has trailing bytes".into());
    }
    Ok((pairs, stats))
}

/// Scored pairs as bytes (to save a run in progress): a count, then for each pair the
/// reference margin, the prompt length, and the full chosen and rejected id lists.
pub fn encode_pairs(pairs: &[Pair]) -> Vec<u8> {
    let mut out = Vec::new();
    put(&mut out, pairs.len() as u32);
    for p in pairs {
        put(&mut out, p.ref_margin.to_bits());
        for len in [p.chosen.prompt_len, p.chosen.ids.len(), p.rejected.ids.len()] {
            put(&mut out, len as u32);
        }
        put_ids(&mut out, &p.chosen.ids);
        put_ids(&mut out, &p.rejected.ids);
    }
    out
}

pub fn decode_pairs(bytes: &[u8]) -> Result<Vec<Pair>, String> {
    let mut r = Reader { bytes, at: 0 };
    let n = r.u32()? as usize;
    let mut pairs = Vec::new();
    for _ in 0..n {
        let ref_margin = r.f32()?;
        let (prompt_len, nc, nr) = (r.u32()? as usize, r.u32()? as usize, r.u32()? as usize);
        if prompt_len == 0 || prompt_len >= nc || prompt_len >= nr {
            return Err("a pair has an empty prompt or an empty answer".into());
        }
        let (chosen, rejected) = (r.ids(nc)?, r.ids(nr)?);
        pairs.push(Pair { chosen: TokenizedExample { ids: chosen, prompt_len }, rejected: TokenizedExample { ids: rejected, prompt_len }, ref_margin });
    }
    if r.at != bytes.len() {
        return Err("pair data has trailing bytes".into());
    }
    Ok(pairs)
}

fn example(prompt_ids: &[usize], response: &[usize]) -> TokenizedExample {
    let mut ids = prompt_ids.to_vec();
    ids.extend_from_slice(response);
    TokenizedExample { ids, prompt_len: prompt_ids.len() }
}

// ---- pairs the trainer can use -----------------------------------------------------------

/// A preference pair plus the reference model's opinion of it.
#[derive(Debug, Clone)]
pub struct Pair {
    pub chosen: TokenizedExample,
    pub rejected: TokenizedExample,
    /// lp_ref(chosen) - lp_ref(rejected), measured once with the frozen reference model.
    pub ref_margin: f32,
}

/// Score mined pairs with the reference model (the model as it is BEFORE any DPO step).
pub fn build_pairs(reference: &Transformer, raw: &[RawPair], threads: usize) -> Vec<Pair> {
    let n = raw.len();
    let seq = reference.cfg.ctx;
    let mut examples: Vec<TokenizedExample> = raw.iter().map(|r| example(&r.prompt_ids, &r.chosen)).collect();
    examples.extend(raw.iter().map(|r| example(&r.prompt_ids, &r.rejected)));
    let all: Vec<usize> = (0..2 * n).collect();
    let (x, y) = pack(&examples, &all, seq);
    let lp = reference.sequence_logprobs(&x, &y, seq, threads);
    (0..n)
        .map(|i| Pair { chosen: examples[i].clone(), rejected: examples[n + i].clone(), ref_margin: lp[i] - lp[n + i] })
        .collect()
}

// ---- training ------------------------------------------------------------------------------

pub struct DpoConfig {
    pub batch: usize,
    pub base_lr: f32,
    /// How strongly to prefer chosen over rejected, relative to the reference.
    pub beta: f32,
    /// Weight of an extra supervised term, `NLL(chosen) / length`, added to the DPO loss.
    /// Plain DPO (0.0) can lower the likelihood of the chosen answers too ("likelihood
    /// displacement"), so probability leaks to nonsense; this term holds them in place.
    pub sft_weight: f32,
    pub weight_decay: f32,
    pub warmup: usize,
    pub clip: f32,
    pub threads: usize,
}

/// Averages over a set of pairs.
#[derive(Debug, Clone, Copy)]
pub struct DpoStats {
    pub loss: f32,
    /// Fraction of pairs where the model now favours the chosen answer MORE than the reference did.
    pub accuracy: f32,
    /// Mean of z: how far, on average, the model has moved toward the chosen answers.
    pub margin: f32,
}

pub struct DpoTrainer {
    pub model: Transformer,
    pub opt: AdamW,
    pub cfg: DpoConfig,
    pub step: usize,
    pub total_steps: usize,
    train: Vec<Pair>,
    val: Vec<Pair>,
    rng: Rng,
}

impl DpoTrainer {
    /// `model` must be the SAME model the pairs were scored with (it is the starting
    /// point; the frozen reference is already baked into each pair's `ref_margin`).
    pub fn new(model: Transformer, cfg: DpoConfig, train: Vec<Pair>, val: Vec<Pair>, total_steps: usize, seed: u64) -> Result<Self, String> {
        if train.is_empty() || val.is_empty() {
            return Err("need at least one training pair and one validation pair".into());
        }
        let opt = AdamW::new(&model.params, cfg.base_lr, cfg.weight_decay);
        Ok(DpoTrainer { model, opt, cfg, step: 0, total_steps, train, val, rng: Rng::new(seed) })
    }

    /// Continue a saved run: set the step counter and reseed the batch stream from it.
    pub fn resume_at(&mut self, step: usize, seed: u64) {
        self.step = step;
        self.rng = Rng::new(seed.wrapping_add((step as u64).wrapping_mul(7919)));
    }

    /// Loss, gradients and statistics for a set of pairs, each pair's loss divided by `norm`
    /// (the size of the WHOLE batch, so that shares of a batch add up).
    fn grads_for(&self, pairs: &[&Pair], norm: usize) -> (f32, Vec<Matrix>, DpoStats) {
        let m = pairs.len();
        if m == 0 {
            return (0.0, self.model.zero_grads(), DpoStats { loss: 0.0, accuracy: 0.0, margin: 0.0 });
        }
        let seq = self.model.cfg.ctx;
        let mut examples: Vec<TokenizedExample> = pairs.iter().map(|p| p.chosen.clone()).collect();
        examples.extend(pairs.iter().map(|p| p.rejected.clone()));
        let all: Vec<usize> = (0..2 * m).collect();
        let (x, y) = pack(&examples, &all, seq);
        let lp = self.model.sequence_logprobs(&x, &y, seq, self.cfg.threads);

        let (beta, inv) = (self.cfg.beta, 1.0 / norm as f32);
        let mut scales = vec![0.0f32; 2 * m];
        let (mut loss, mut correct, mut z_sum) = (0.0, 0usize, 0.0);
        for i in 0..m {
            let z = beta * ((lp[i] - lp[m + i]) - pairs[i].ref_margin);
            loss += softplus_neg(z) * inv;
            correct += (z > 0.0) as usize;
            z_sum += z;
            let w = beta * sigmoid_neg(z) * inv;
            // The optional supervised term: alpha * NLL(chosen) / (number of answer tokens).
            let chosen_len = y[i * seq..(i + 1) * seq].iter().filter(|&&t| t != crate::transformer::IGNORE_TARGET).count().max(1);
            scales[i] = w + self.cfg.sft_weight * inv / chosen_len as f32; // + w * grad NLL(chosen):  pulls it up
            scales[m + i] = -w; //                                           - w * grad NLL(rejected): pushes it down
        }
        let grads = self.model.weighted_grads(&x, &y, seq, &scales, self.cfg.threads);
        let stats = DpoStats { loss: loss * norm as f32 / m as f32, accuracy: correct as f32 / m as f32, margin: z_sum / m as f32 };
        (loss, grads, stats)
    }

    /// Draw this step's batch (every data-parallel worker draws the same one).
    fn draw_batch(&mut self) -> Vec<usize> {
        (0..self.cfg.batch).map(|_| self.rng.next_usize(self.train.len())).collect()
    }

    pub fn train_step(&mut self) -> DpoStats {
        let batch = self.draw_batch();
        let pairs: Vec<&Pair> = batch.iter().map(|&i| &self.train[i]).collect();
        let (_, mut grads, stats) = self.grads_for(&pairs, self.cfg.batch);
        self.apply_grads(&mut grads);
        stats
    }

    /// Data-parallel half 1 (see `parallel`): gradients for worker `shard`'s share of the batch.
    pub fn compute_grads(&mut self, shard: usize, n_shards: usize) -> (f32, Vec<Matrix>) {
        let batch = self.draw_batch();
        let mine = shard_range(self.cfg.batch, shard, n_shards);
        let pairs: Vec<&Pair> = batch[mine].iter().map(|&i| &self.train[i]).collect();
        let (loss, grads, _) = self.grads_for(&pairs, self.cfg.batch);
        (loss, grads)
    }

    /// Data-parallel half 2: apply the summed gradients.
    pub fn apply_grads(&mut self, grads: &mut [Matrix]) {
        clip_grad_norm(grads, self.cfg.clip);
        self.opt.lr = lr_at(self.step, self.total_steps, self.cfg.warmup, self.cfg.base_lr);
        self.opt.step(&mut self.model.params, grads);
        self.step += 1;
    }

    /// DPO statistics on held-out pairs.
    pub fn validate(&self) -> DpoStats {
        evaluate_pairs(&self.model, &self.val[..self.val.len().min(256)], self.cfg.beta, self.cfg.threads)
    }
}

/// Statistics of a model on pairs (no training).
pub fn evaluate_pairs(model: &Transformer, pairs: &[Pair], beta: f32, threads: usize) -> DpoStats {
    let m = pairs.len();
    let seq = model.cfg.ctx;
    let mut examples: Vec<TokenizedExample> = pairs.iter().map(|p| p.chosen.clone()).collect();
    examples.extend(pairs.iter().map(|p| p.rejected.clone()));
    let all: Vec<usize> = (0..2 * m).collect();
    let (x, y) = pack(&examples, &all, seq);
    let lp = model.sequence_logprobs(&x, &y, seq, threads);
    let (mut loss, mut correct, mut z_sum) = (0.0, 0usize, 0.0);
    for i in 0..m {
        let z = beta * ((lp[i] - lp[m + i]) - pairs[i].ref_margin);
        loss += softplus_neg(z);
        correct += (z > 0.0) as usize;
        z_sum += z;
    }
    let n = m.max(1) as f32;
    DpoStats { loss: loss / n, accuracy: correct as f32 / n, margin: z_sum / n }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TransformerConfig;

    fn model() -> Transformer {
        let cfg = TransformerConfig { vocab_size: 9, ctx: 12, d_model: 16, n_layer: 2, n_head: 2 };
        Transformer::new(cfg, &mut Rng::new(21))
    }

    /// Pairs on a toy task: after the prompt [s, 7], the chosen answer repeats the symbol
    /// s three times then stops (token 8); the rejected answer repeats a different symbol.
    fn toy_pairs(reference: &Transformer, n: usize) -> Vec<Pair> {
        let raw: Vec<RawPair> = (0..n)
            .map(|i| {
                let s = 1 + i % 4;
                RawPair { prompt_ids: vec![s, 7], chosen: vec![s, s, s, 8], rejected: vec![1 + (i + 1) % 4, 1 + (i + 2) % 4, 5, 8] }
            })
            .collect();
        build_pairs(reference, &raw, 1)
    }

    fn cfg(batch: usize) -> DpoConfig {
        DpoConfig { batch, base_lr: 3e-3, beta: 0.5, sft_weight: 0.0, weight_decay: 0.0, warmup: 2, clip: 5.0, threads: 1 }
    }

    #[test]
    fn numeric_helpers_are_stable_and_correct() {
        assert!((softplus_neg(0.0) - 2.0f32.ln()).abs() < 1e-6);
        assert!((sigmoid_neg(0.0) - 0.5).abs() < 1e-6);
        // Extreme inputs must not overflow to inf/NaN.
        for z in [-200.0f32, -30.0, 30.0, 200.0] {
            assert!(softplus_neg(z).is_finite() && sigmoid_neg(z).is_finite(), "z = {z}");
        }
        assert!((softplus_neg(200.0)).abs() < 1e-6); // confident and right: no loss
        assert!((softplus_neg(-200.0) - 200.0).abs() < 1e-3); // confident and wrong: loss ~ |z|
        // d/dz softplus_neg = -sigmoid(-z)
        let (z, h) = (0.7f32, 1e-3);
        let numeric = (softplus_neg(z + h) - softplus_neg(z - h)) / (2.0 * h);
        assert!((numeric + sigmoid_neg(z)).abs() < 1e-3);
    }

    #[test]
    fn at_the_start_the_policy_equals_the_reference() {
        let m = model();
        let pairs = toy_pairs(&m, 8);
        let stats = evaluate_pairs(&m, &pairs, 0.5, 1);
        assert!((stats.loss - 2.0f32.ln()).abs() < 1e-5, "loss {}", stats.loss);
        assert!(stats.margin.abs() < 1e-5);
        assert_eq!(stats.accuracy, 0.0); // z is exactly 0, which is not "favouring the chosen answer"
    }

    #[test]
    fn dpo_gradient_matches_numerical_estimate_of_the_true_loss() {
        // The decisive test: differentiate the ACTUAL DPO loss numerically and compare
        // with the hand-derived "weighted cross-entropy" gradient.
        let reference = model();
        let mut pairs = toy_pairs(&reference, 5);
        for (i, p) in pairs.iter_mut().enumerate() {
            p.ref_margin += [0.4, -0.3, 0.1, 0.0, -0.6][i]; // pretend the reference disagreed with itself
        }
        let mut policy = model();
        // Move the policy away from the reference so z is not 0 and the weights are non-trivial.
        for (k, p) in policy.params.iter_mut().enumerate() {
            for (j, w) in p.data.iter_mut().enumerate() {
                *w += 0.03 * ((j * 7 + k * 3) as f32).sin();
            }
        }
        let dpo_loss = |m: &Transformer| evaluate_pairs(m, &pairs, 0.5, 1).loss;

        let trainer = DpoTrainer::new(policy, cfg(5), pairs.clone(), pairs.clone(), 10, 1).unwrap();
        let refs: Vec<&Pair> = pairs.iter().collect();
        let (loss, grads, _) = trainer.grads_for(&refs, 5);
        let mut policy = trainer.model;
        assert!((loss - dpo_loss(&policy)).abs() < 1e-4, "{loss} vs {}", dpo_loss(&policy));

        let h = 1e-2;
        let mut checked = 0;
        for p in 0..policy.params.len() {
            for idx in (0..policy.params[p].data.len()).step_by(11) {
                let original = policy.params[p].data[idx];
                policy.params[p].data[idx] = original + h;
                let up = dpo_loss(&policy);
                policy.params[p].data[idx] = original - h;
                let down = dpo_loss(&policy);
                policy.params[p].data[idx] = original;
                let numeric = (up - down) / (2.0 * h);
                let analytic = grads[p].data[idx];
                assert!((numeric - analytic).abs() < 2e-3 + 0.05 * analytic.abs(), "param {p} entry {idx}: numeric {numeric} vs analytic {analytic}");
                checked += 1;
            }
        }
        assert!(checked > 100);
    }

    #[test]
    fn the_supervised_term_has_the_right_gradient_and_holds_chosen_answers_up() {
        let reference = model();
        let pairs = toy_pairs(&reference, 5);
        let mut policy = model();
        for (k, p) in policy.params.iter_mut().enumerate() {
            for (j, w) in p.data.iter_mut().enumerate() {
                *w += 0.03 * ((j * 5 + k) as f32).cos();
            }
        }
        // Objective = DPO loss + alpha * mean over pairs of (NLL(chosen) / answer length).
        let alpha = 0.8;
        let objective = |m: &Transformer| -> f32 {
            let dpo = evaluate_pairs(m, &pairs, 0.5, 1).loss;
            let mut nll = 0.0;
            for p in &pairs {
                let (x, y) = pack(std::slice::from_ref(&p.chosen), &[0], m.cfg.ctx);
                let len = y.iter().filter(|&&t| t != crate::transformer::IGNORE_TARGET).count() as f32;
                nll += -m.sequence_logprobs(&x, &y, m.cfg.ctx, 1)[0] / len;
            }
            dpo + alpha * nll / pairs.len() as f32
        };
        let mut config = cfg(5);
        config.sft_weight = alpha;
        let trainer = DpoTrainer::new(policy, config, pairs.clone(), pairs.clone(), 10, 1).unwrap();
        let refs: Vec<&Pair> = pairs.iter().collect();
        let (_, grads, _) = trainer.grads_for(&refs, 5);
        let mut policy = trainer.model;
        let h = 1e-2;
        for p in 0..policy.params.len() {
            for idx in (0..policy.params[p].data.len()).step_by(13) {
                let original = policy.params[p].data[idx];
                policy.params[p].data[idx] = original + h;
                let up = objective(&policy);
                policy.params[p].data[idx] = original - h;
                let down = objective(&policy);
                policy.params[p].data[idx] = original;
                let numeric = (up - down) / (2.0 * h);
                assert!((numeric - grads[p].data[idx]).abs() < 2e-3 + 0.05 * grads[p].data[idx].abs(), "param {p} entry {idx}: {numeric} vs {}", grads[p].data[idx]);
            }
        }
    }

    #[test]
    fn training_moves_the_model_toward_the_chosen_answers() {
        let reference = model();
        let pairs = toy_pairs(&reference, 40);
        let (train, val) = (pairs[..32].to_vec(), pairs[32..].to_vec());
        let mut t = DpoTrainer::new(reference, cfg(8), train, val, 120, 5).unwrap();
        let before = t.validate();
        for _ in 0..120 {
            t.train_step();
        }
        let after = t.validate();
        assert!(after.loss < before.loss * 0.5, "loss {} -> {}", before.loss, after.loss);
        assert!(after.accuracy > 0.9, "accuracy {}", after.accuracy);
        assert!(after.margin > before.margin);
    }

    #[test]
    fn parallel_workers_match_single_process_dpo() {
        use crate::parallel::{add_flat, checksum, flat_to_grads, grads_to_flat};
        let make = || {
            let reference = model();
            let pairs = toy_pairs(&reference, 30);
            DpoTrainer::new(reference, cfg(7), pairs[..24].to_vec(), pairs[24..].to_vec(), 40, 3).unwrap()
        };
        let mut solo = make();
        let mut pool: Vec<DpoTrainer> = (0..3).map(|_| make()).collect();
        for step in 0..6 {
            let solo_loss = solo.train_step().loss;
            let mut total: Option<Vec<f32>> = None;
            for (i, r) in pool.iter_mut().enumerate() {
                let (loss, grads) = r.compute_grads(i, 3);
                let flat = grads_to_flat(&grads, loss);
                match &mut total {
                    None => total = Some(flat),
                    Some(t) => add_flat(t, &flat),
                }
            }
            let total = total.unwrap();
            for r in pool.iter_mut() {
                let (mut g, _) = flat_to_grads(&r.model.params, &total).unwrap();
                r.apply_grads(&mut g);
            }
            assert!((solo_loss - total.last().unwrap()).abs() < 1e-4, "step {step}");
        }
        assert_eq!(checksum(&pool[0].model.params), checksum(&pool[2].model.params));
        for (a, b) in solo.model.params.iter().zip(&pool[0].model.params) {
            for (x, y) in a.data.iter().zip(&b.data) {
                assert!((x - y).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn pairs_survive_a_trip_through_bytes() {
        let reference = model();
        let mut pairs = toy_pairs(&reference, 6);
        pairs[2].ref_margin = -1.25; // an exact f32 must come back bit for bit
        let back = decode_pairs(&encode_pairs(&pairs)).unwrap();
        assert_eq!(back.len(), 6);
        for (a, b) in pairs.iter().zip(&back) {
            assert_eq!((a.ref_margin.to_bits(), &a.chosen.ids, a.chosen.prompt_len), (b.ref_margin.to_bits(), &b.chosen.ids, b.chosen.prompt_len));
            assert_eq!((&a.rejected.ids, a.rejected.prompt_len), (&b.rejected.ids, b.rejected.prompt_len));
        }

        let raw = vec![
            RawPair { prompt_ids: vec![1, 2], chosen: vec![3, 4, 5], rejected: vec![6] },
            RawPair { prompt_ids: vec![9], chosen: vec![8], rejected: vec![7, 7] },
        ];
        let stats = MineStats { prompts: 10, samples: 40, good: 25, pairs: 2 };
        let (raw_back, stats_back) = decode_mined(&encode_mined(&raw, &stats)).unwrap();
        assert_eq!((stats_back.prompts, stats_back.samples, stats_back.good, stats_back.pairs), (10, 40, 25, 2));
        assert_eq!(raw_back.len(), 2);
        assert_eq!((&raw_back[1].prompt_ids, &raw_back[1].chosen, &raw_back[1].rejected), (&raw[1].prompt_ids, &raw[1].chosen, &raw[1].rejected));
    }

    #[test]
    fn corrupt_pair_data_is_rejected_without_panicking_or_huge_allocations() {
        let reference = model();
        let good = encode_pairs(&toy_pairs(&reference, 3));
        assert!(decode_pairs(&good[..good.len() - 1]).is_err(), "truncated");
        let mut extra = good.clone();
        extra.push(0);
        assert!(decode_pairs(&extra).is_err(), "trailing byte");
        assert!(decode_pairs(&[]).is_err());
        // A header that claims 4 billion pairs / a 4-billion-id answer.
        assert!(decode_pairs(&u32::MAX.to_le_bytes()).is_err());
        let mut lie = Vec::new();
        for v in [1u32, 0, 1, u32::MAX, 2] {
            lie.extend_from_slice(&v.to_le_bytes());
        }
        assert!(decode_pairs(&lie).is_err());
        assert!(decode_mined(&[1, 2, 3]).is_err());
    }

    #[test]
    fn a_resumed_trainer_continues_with_fresh_batches() {
        let reference = model();
        let pairs = toy_pairs(&reference, 30);
        let make = |step: usize| {
            let mut t = DpoTrainer::new(model(), cfg(6), pairs[..24].to_vec(), pairs[24..].to_vec(), 50, 9).unwrap();
            t.resume_at(step, 9);
            t
        };
        // At the start every pair has z = 0, so the LOSS is ln 2 whatever the batch; the
        // gradients do depend on which pairs were drawn, so compare those.
        let (_, ga) = make(0).compute_grads(0, 1);
        let (_, gb) = make(20).compute_grads(0, 1);
        let differs = ga.iter().zip(&gb).any(|(x, y)| x.data != y.data);
        assert!(differs, "a resumed run must not replay the first batches");
        let (_, gc) = make(20).compute_grads(0, 1);
        assert!(gb.iter().zip(&gc).all(|(x, y)| x.data == y.data), "resuming at the same step is deterministic");
        assert_eq!(make(20).step, 20);
    }

    #[test]
    fn pairing_needs_one_good_and_one_bad_answer() {
        assert_eq!(pick_pair(&[false, true, true, false]), Some((1, 0)));
        assert_eq!(pick_pair(&[true, true, true]), None);
        assert_eq!(pick_pair(&[false, false]), None);
        assert_eq!(pick_pair(&[]), None);
    }

    #[test]
    fn word_of_reads_the_prompt() {
        assert_eq!(word_of("Word: dragon\nSentence:"), "dragon");
        assert!(uses_word_and_stops("A big dragon.", true, "dragon"));
        assert!(!uses_word_and_stops("A big dragon.", false, "dragon"), "must finish properly");
        assert!(!uses_word_and_stops("A big cat.", true, "dragon"));
    }

    #[test]
    fn mining_is_deterministic_whatever_the_thread_count() {
        let text = "Word: ball\nSentence: A red ball.<|endoftext|> the cat sat on the mat. ".repeat(30);
        let tok = Tokenizer::train(&text, 300);
        let cfg = TransformerConfig { vocab_size: tok.vocab_size(), ctx: 32, d_model: 16, n_layer: 1, n_head: 2 };
        let m = Transformer::new(cfg, &mut Rng::new(4));
        let prompts: Vec<String> = ["ball", "cat", "mat", "red", "sat", "dog"].iter().map(|w| format!("Word: {w}\nSentence:")).collect();
        // A deterministic but arbitrary verifier, so that both good and bad answers occur.
        let verify = |text: &str, _stopped: bool, _word: &str| text.len() % 2 == 0;
        let (a, sa) = mine_pairs(&m, &tok, &prompts, 0, 6, 1.0, 12, 77, 1, &verify);
        let (b, sb) = mine_pairs(&m, &tok, &prompts, 0, 6, 1.0, 12, 77, 4, &verify);

        // Mining the list in two pieces (as two browser workers do) and joining the pieces
        // in order gives exactly the same pairs as mining it all at once.
        let (left, sl) = mine_pairs(&m, &tok, &prompts[..2], 0, 6, 1.0, 12, 77, 1, &verify);
        let (right, sr) = mine_pairs(&m, &tok, &prompts[2..], 2, 6, 1.0, 12, 77, 1, &verify);
        let joined: Vec<_> = left.iter().chain(&right).collect();
        assert_eq!(joined.len(), a.len());
        assert_eq!((sl.samples + sr.samples, sl.good + sr.good, sl.pairs + sr.pairs), (sa.samples, sa.good, sa.pairs));
        for (p, q) in joined.iter().zip(&a) {
            assert_eq!((&p.chosen, &p.rejected, &p.prompt_ids), (&q.chosen, &q.rejected, &q.prompt_ids));
        }
        assert_eq!((sa.samples, sa.good, sa.pairs), (sb.samples, sb.good, sb.pairs));
        assert_eq!(sa.samples, 36);
        assert!(sa.pairs > 0, "the arbitrary verifier should split some prompts: {sa:?}");
        for (p, q) in a.iter().zip(&b) {
            assert_eq!((&p.chosen, &p.rejected), (&q.chosen, &q.rejected));
        }
        // Each mined pair obeys the verifier exactly: the chosen answer's text (cut at the stop
        // marker) has even length, the rejected one's has odd length.
        let text_of = |ids: &[usize]| {
            let full = tok.decode(ids);
            full.find(STOP).map_or(full.clone(), |at| full[..at].to_string())
        };
        for p in &a {
            assert_eq!(text_of(&p.chosen).len() % 2, 0, "chosen must be good");
            assert_eq!(text_of(&p.rejected).len() % 2, 1, "rejected must be bad");
            assert!(p.prompt_ids.len() + p.chosen.len() < 32 && p.prompt_ids.len() + p.rejected.len() < 32);
        }
    }
}
