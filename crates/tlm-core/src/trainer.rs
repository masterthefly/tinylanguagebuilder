//! The training loop as a reusable object: one call = one optimisation step.
//! The command-line tool drives it today; a browser UI can drive it later
//! (a few steps per animation frame, so the page stays responsive).

use crate::optim::{clip_grad_norm, AdamW};
use crate::parallel::shard_range;
use crate::transformer::Transformer;
use crate::{Matrix, Rng};

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub seq: usize,
    pub batch: usize,
    pub base_lr: f32,
    pub weight_decay: f32,
    pub warmup: usize,
    pub clip: f32,
    pub threads: usize,
}

pub struct Trainer {
    pub model: Transformer,
    pub opt: AdamW,
    pub cfg: TrainConfig,
    /// Steps completed so far.
    pub step: usize,
    /// Where the learning-rate schedule ends.
    pub total_steps: usize,
    rng: Rng,
    train_ids: Vec<usize>,
    val_x: Vec<usize>,
    val_y: Vec<usize>,
}

/// Linear warm-up for `warmup` steps, then cosine decay down to 10% of `base`.
pub fn lr_at(step: usize, total: usize, warmup: usize, base: f32) -> f32 {
    if step < warmup {
        return base * (step + 1) as f32 / warmup as f32;
    }
    let span = total.saturating_sub(warmup).max(1) as f32;
    let progress = ((step - warmup) as f32 / span).min(1.0);
    base * (0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos()))
}

/// `n` random windows: input = ids[s..s+seq], target = the same window shifted by one.
fn make_batch(ids: &[usize], seq: usize, n: usize, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
    let (mut x, mut y) = (Vec::with_capacity(n * seq), Vec::with_capacity(n * seq));
    for _ in 0..n {
        let s = rng.next_usize(ids.len() - seq - 1);
        x.extend_from_slice(&ids[s..s + seq]);
        y.extend_from_slice(&ids[s + 1..s + seq + 1]);
    }
    (x, y)
}

impl Trainer {
    /// `ids` is the whole tokenised corpus: the first 90% trains, the last 10%
    /// is held out for validation. The split depends only on `ids`, so a
    /// resumed run validates on exactly the same text as the original.
    pub fn new(model: Transformer, opt: AdamW, cfg: TrainConfig, ids: &[usize], step: usize, total_steps: usize, seed: u64) -> Result<Self, String> {
        let split = ids.len() * 9 / 10;
        let (train_ids, val_ids) = (ids[..split].to_vec(), &ids[split..]);
        let min_len = cfg.seq + 2;
        if train_ids.len() < min_len || val_ids.len() < min_len {
            return Err(format!("not enough text: need at least {min_len} tokens in each of train and validation"));
        }
        if cfg.seq > model.cfg.ctx {
            return Err(format!("sequence length {} exceeds the model's context {}", cfg.seq, model.cfg.ctx));
        }
        let (val_x, val_y) = make_batch(val_ids, cfg.seq, 64, &mut Rng::new(999));
        // Mix the step into the seed so a resumed run draws NEW batches
        // instead of replaying the ones it already trained on.
        let rng = Rng::new(seed.wrapping_add((step as u64).wrapping_mul(7919)));
        Ok(Trainer { model, opt, cfg, step, total_steps, rng, train_ids, val_x, val_y })
    }

    /// Learning rate for the current step: linear warm-up, then cosine decay to 10%.
    pub fn learning_rate(&self) -> f32 {
        lr_at(self.step, self.total_steps, self.cfg.warmup, self.cfg.base_lr)
    }

    /// One optimisation step on a random batch. Returns the batch's training loss.
    pub fn train_step(&mut self) -> f32 {
        let (loss, mut grads) = self.compute_grads(0, 1);
        self.apply_grads(&mut grads);
        loss
    }

    /// Data-parallel half 1: draw the WHOLE batch (so every worker's random
    /// stream stays in step) but compute gradients only for worker `shard`'s
    /// share. The returned loss and gradients are already divided by the whole
    /// batch's size, so the shares of all workers simply ADD UP to the full
    /// result. Does not change the model or the step counter.
    pub fn compute_grads(&mut self, shard: usize, n_shards: usize) -> (f32, Vec<Matrix>) {
        let seq = self.cfg.seq;
        let (x, y) = make_batch(&self.train_ids, seq, self.cfg.batch, &mut self.rng);
        let mine = shard_range(self.cfg.batch, shard, n_shards);
        let (xs, ys) = (&x[mine.start * seq..mine.end * seq], &y[mine.start * seq..mine.end * seq]);
        self.model.loss_and_grads_scaled(xs, ys, seq, self.cfg.threads, self.cfg.batch * seq)
    }

    /// Data-parallel half 2: apply gradients (the sum over all workers' shares)
    /// with clipping and the optimiser, and advance the step counter. Every
    /// worker calls this with identical input, so their models stay identical.
    pub fn apply_grads(&mut self, grads: &mut [Matrix]) {
        clip_grad_norm(grads, self.cfg.clip);
        self.opt.lr = self.learning_rate();
        self.opt.step(&mut self.model.params, grads);
        self.step += 1;
    }

    /// Loss on the fixed held-out windows.
    pub fn validate(&self) -> f32 {
        self.model.loss(&self.val_x, &self.val_y, self.cfg.seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TransformerConfig;

    fn trainer(step: usize) -> Trainer {
        let cfg = TransformerConfig { vocab_size: 6, ctx: 8, d_model: 16, n_layer: 1, n_head: 2 };
        let model = Transformer::new(cfg, &mut Rng::new(1));
        let opt = AdamW::new(&model.params, 3e-3, 0.0);
        let ids: Vec<usize> = (0..400).map(|i| i % 6).collect();
        let tc = TrainConfig { seq: 8, batch: 4, base_lr: 3e-3, weight_decay: 0.0, warmup: 10, clip: 1.0, threads: 1 };
        Trainer::new(model, opt, tc, &ids, step, 100, 7).unwrap()
    }

    #[test]
    fn learning_rate_warms_up_then_decays() {
        let mut t = trainer(0);
        let start = t.learning_rate();
        t.step = 10;
        let peak = t.learning_rate();
        t.step = 100;
        let end = t.learning_rate();
        assert!(start < peak && end < peak);
        assert!((end - 3e-4).abs() < 1e-6, "ends at 10% of peak, got {end}");
    }

    #[test]
    fn training_a_repeating_pattern_lowers_validation_loss() {
        let mut t = trainer(0);
        let before = t.validate();
        for _ in 0..60 {
            t.train_step();
        }
        assert_eq!(t.step, 60);
        assert!(t.validate() < before * 0.7, "{before} -> {}", t.validate());
    }

    /// Run one data-parallel step across `replicas` (all identical), as the
    /// browser's worker pool does: each computes its share, shares are summed,
    /// and everyone applies the same total. Returns the step's total loss.
    fn parallel_step(replicas: &mut [Trainer]) -> f32 {
        use crate::parallel::{add_flat, flat_to_grads, grads_to_flat};
        let n = replicas.len();
        let mut total: Option<Vec<f32>> = None;
        for (i, r) in replicas.iter_mut().enumerate() {
            let (loss, grads) = r.compute_grads(i, n);
            let flat = grads_to_flat(&grads, loss);
            match &mut total {
                None => total = Some(flat),
                Some(t) => add_flat(t, &flat),
            }
        }
        let total = total.unwrap();
        for r in replicas.iter_mut() {
            let (mut grads, _) = flat_to_grads(&r.model.params, &total).unwrap();
            r.apply_grads(&mut grads);
        }
        *total.last().unwrap()
    }

    #[test]
    fn data_parallel_training_matches_single_process_training() {
        for n_workers in [2, 3, 4] {
            let mut solo = trainer(0);
            let mut pool: Vec<Trainer> = (0..n_workers).map(|_| trainer(0)).collect();
            for step in 0..12 {
                let solo_loss = solo.train_step();
                let pool_loss = parallel_step(&mut pool);
                assert!((solo_loss - pool_loss).abs() < 1e-4, "{n_workers} workers, step {step}: {solo_loss} vs {pool_loss}");
            }
            // The workers never drift apart: bit-for-bit identical...
            for r in &pool[1..] {
                assert_eq!(crate::parallel::checksum(&r.model.params), crate::parallel::checksum(&pool[0].model.params));
            }
            // ...and they agree with single-process training up to floating-point summation order.
            for (a, b) in solo.model.params.iter().zip(&pool[0].model.params) {
                for (x, y) in a.data.iter().zip(&b.data) {
                    assert!((x - y).abs() < 1e-3, "{n_workers} workers: weight {x} vs {y}");
                }
            }
        }
    }

    #[test]
    fn more_workers_than_sequences_still_works() {
        // batch is 4, so with 6 workers some get an empty share.
        let mut pool: Vec<Trainer> = (0..6).map(|_| trainer(0)).collect();
        let mut solo = trainer(0);
        let (a, b) = (solo.train_step(), parallel_step(&mut pool));
        assert!((a - b).abs() < 1e-4);
    }

    #[test]
    fn resumed_runs_see_different_batches() {
        let mut a = trainer(0);
        let mut b = trainer(50);
        assert_ne!(a.train_step(), b.train_step());
    }

    #[test]
    fn too_little_text_is_an_error() {
        let cfg = TransformerConfig { vocab_size: 6, ctx: 8, d_model: 16, n_layer: 1, n_head: 2 };
        let model = Transformer::new(cfg, &mut Rng::new(1));
        let opt = AdamW::new(&model.params, 1e-3, 0.0);
        let tc = TrainConfig { seq: 8, batch: 4, base_lr: 1e-3, weight_decay: 0.0, warmup: 1, clip: 1.0, threads: 1 };
        assert!(Trainer::new(model, opt, tc, &[0, 1, 2, 3], 0, 10, 1).is_err());
    }
}
