//! Train a BPE tokenizer and the MLP on real TinyStories text.
//! Run with:  cargo run --release --example tinystories_demo
//! Optional:  cargo run --release --example tinystories_demo -- path/to/text.txt

use std::time::Instant;
use tlm_core::bpe::Tokenizer;
use tlm_core::mlp::{make_dataset, sample_batch, Mlp, MlpConfig};
use tlm_core::Rng;

const TEXT_BYTES: usize = 2_000_000; // use the first 2 MB for now
const VOCAB_SIZE: usize = 512;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "data/tinystories/valid.txt".to_string());
    let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let text = String::from_utf8_lossy(&raw[..TEXT_BYTES.min(raw.len())]).into_owned();
    println!("text: {} bytes from {path}", text.len());

    // ---- 1. Tokenizer ----
    let t0 = Instant::now();
    let tok = Tokenizer::train(&text, VOCAB_SIZE);
    println!("trained BPE ({} tokens) in {:.1}s", tok.vocab_size(), t0.elapsed().as_secs_f32());

    println!("\nfirst merges (the most common pieces of English):");
    for k in 0..8 {
        println!("  {:>3} = {:?}", 256 + k, String::from_utf8_lossy(&tok.vocab[256 + k]));
    }
    println!("last merges:");
    for k in (tok.merges.len() - 5)..tok.merges.len() {
        println!("  {:>3} = {:?}", 256 + k, String::from_utf8_lossy(&tok.vocab[256 + k]));
    }

    let sample = "Once upon a time, a little girl named Lily had a red ball.";
    let ids = tok.encode(sample);
    println!("\n\"{sample}\"");
    println!("{} bytes -> {} tokens: {:?}", sample.len(), ids.len(), ids);
    let pieces: Vec<String> = ids.iter().map(|&i| tok.decode(&[i])).collect();
    println!("{:?}", pieces);

    let ids = tok.encode(&text);
    let bytes_per_token = text.len() as f32 / ids.len() as f32;
    println!("\nwhole text: {} bytes -> {} tokens ({bytes_per_token:.2} bytes/token)", text.len(), ids.len());

    // ---- 2. Model ----
    let split = ids.len() * 9 / 10;
    let cfg = MlpConfig { vocab_size: tok.vocab_size(), context: 8, emb_dim: 16, hidden: 128 };
    let (train_ctx, train_t) = make_dataset(&ids[..split], cfg.context);
    let (val_ctx, val_t) = make_dataset(&ids[split..], cfg.context);
    println!("train examples: {}   validation examples: {}", train_t.len(), val_t.len());

    let mut rng = Rng::new(7);
    let mut model = Mlp::new(cfg, &mut rng);
    println!("MLP parameters: {}\n", model.num_params());

    // Loss in "bits per byte" is comparable across tokenizers: 8.0 = clueless.
    let to_bpb = |loss: f32| loss / std::f32::consts::LN_2 / bytes_per_token;

    let steps = 6000;
    let t0 = Instant::now();
    for step in 0..=steps {
        let (bx, by) = sample_batch(&train_ctx, &train_t, cfg.context, 256, &mut rng);
        let loss = model.train_step(&bx, &by, 0.2);
        if step % 1000 == 0 {
            // Validate on a fixed-size random slice to keep this quick.
            let (vx, vy) = sample_batch(&val_ctx, &val_t, cfg.context, 4000, &mut rng);
            let val = model.loss(&vx, &vy);
            println!(
                "step {step:>5}  train {loss:.3}  val {val:.3}  ({:.2} bits/byte)  [{:.0}s]",
                to_bpb(val),
                t0.elapsed().as_secs_f32()
            );
        }
    }

    println!("\n--- generated text ---");
    let start = tok.encode("Once upon a time, ");
    let start = &start[start.len() - cfg.context.min(start.len())..];
    // Pad if the prompt is shorter than the context window.
    let mut prompt = vec![tok.encode(" ")[0]; cfg.context - start.len()];
    prompt.extend_from_slice(start);
    let generated = model.sample(&prompt, 150, &mut rng);
    println!("Once upon a time, {}", tok.decode(&generated));
}
