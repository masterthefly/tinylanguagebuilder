//! Train a tiny transformer on TinyStories (CPU only, multi-threaded).
//! Run with:  cargo run --release --example transformer_demo
//! Optional:  cargo run --release --example transformer_demo -- <steps>

use std::time::Instant;
use tlm_core::bpe::Tokenizer;
use tlm_core::optim::{clip_grad_norm, AdamW};
use tlm_core::transformer::{Transformer, TransformerConfig};
use tlm_core::Rng;

const TEXT_BYTES: usize = 4_000_000;
const VOCAB_SIZE: usize = 512;
const SEQ: usize = 32;
const BATCH: usize = 16;
const BASE_LR: f32 = 3e-3;

/// Random windows: input = ids[s..s+SEQ], target = the same window shifted by one.
fn make_batch(ids: &[usize], n: usize, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
    let (mut x, mut y) = (Vec::with_capacity(n * SEQ), Vec::with_capacity(n * SEQ));
    for _ in 0..n {
        let s = rng.next_usize(ids.len() - SEQ - 1);
        x.extend_from_slice(&ids[s..s + SEQ]);
        y.extend_from_slice(&ids[s + 1..s + SEQ + 1]);
    }
    (x, y)
}

/// Warm up linearly, then decay along a cosine to 10% of the peak.
fn learning_rate(step: usize, total: usize) -> f32 {
    let warmup = 100;
    if step < warmup {
        return BASE_LR * (step + 1) as f32 / warmup as f32;
    }
    let progress = (step - warmup) as f32 / (total - warmup).max(1) as f32;
    BASE_LR * (0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos()))
}

fn main() {
    let steps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    let raw = std::fs::read("data/tinystories/valid.txt").expect("run from the project root; see Lesson 5 for the data");
    let text = String::from_utf8_lossy(&raw[..TEXT_BYTES.min(raw.len())]).into_owned();
    let tok = Tokenizer::train(&text, VOCAB_SIZE);
    let ids = tok.encode(&text);
    let bytes_per_token = text.len() as f32 / ids.len() as f32;
    let split = ids.len() * 9 / 10;
    let (train_ids, val_ids) = (&ids[..split], &ids[split..]);
    println!("{} tokens ({bytes_per_token:.2} bytes/token), using {threads} threads", ids.len());

    let cfg = TransformerConfig { vocab_size: tok.vocab_size(), ctx: SEQ, d_model: 64, n_layer: 2, n_head: 4 };
    let mut rng = Rng::new(42);
    let mut model = Transformer::new(cfg, &mut rng);
    let mut opt = AdamW::new(&model.params, BASE_LR, 0.01);
    println!("transformer parameters: {}", model.num_params());
    println!("tokens per step: {}\n", BATCH * SEQ);

    let to_bpb = |loss: f32| loss / std::f32::consts::LN_2 / bytes_per_token;
    let (vx, vy) = make_batch(val_ids, 64, &mut Rng::new(999)); // fixed validation set
    let start = Instant::now();

    for step in 0..=steps {
        let (bx, by) = make_batch(train_ids, BATCH, &mut rng);
        let (loss, mut grads) = model.loss_and_grads(&bx, &by, SEQ, threads);
        clip_grad_norm(&mut grads, 1.0);
        opt.lr = learning_rate(step, steps);
        opt.step(&mut model.params, &grads);

        if step % 250 == 0 {
            let val = model.loss(&vx, &vy, SEQ);
            println!(
                "step {step:>5}  train {loss:.3}  val {val:.3}  ({:.2} bits/byte)  [{:.0}s]",
                to_bpb(val),
                start.elapsed().as_secs_f32()
            );
        }
    }

    println!("\n--- generated text (temperature 0.8) ---");
    let prompt = tok.encode("Once upon a time");
    let out = model.sample(&prompt, 120, 0.8, &mut rng);
    println!("Once upon a time{}", tok.decode(&out));
}
