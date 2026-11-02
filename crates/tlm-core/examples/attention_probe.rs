//! What does each attention head of a TRAINED model look at? Averages over many text windows.
//! Run with:  cargo run --release --example attention_probe -- <model-dir> <text-file>
//! e.g.       cargo run --release --example attention_probe -- runs/emb-tiny data/tinystories/valid.txt
//!
//! For every layer and head it reports the average attention weight on: the token itself, the
//! previous token, the first token of the window, and everything else, plus how spread out the
//! attention is (1.0 = perfectly even over the visible tokens, near 0 = all on one token).

use std::path::Path;
use tlm_core::checkpoint::{load_model, load_tokenizer};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: attention_probe <model-dir> <text-file> [windows]");
        std::process::exit(2);
    }
    let windows: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(300);
    let (model, step) = load_model(Path::new(&args[1])).expect("load model");
    let tok = load_tokenizer(Path::new(&args[1])).expect("load tokenizer");
    let text = std::fs::read_to_string(&args[2]).expect("read text");
    // Use text from the far end of the file so it is unlikely to be what the model trained on.
    let tail = &text[text.len().saturating_sub(2_000_000)..];
    let mut start = 0;
    while !tail.is_char_boundary(start) {
        start += 1;
    }
    let ids = tok.encode(&tail[start..]);
    let ctx = model.cfg.ctx;
    println!("model: {} layers, {} heads, d_model {}, trained {} steps; {} windows of {} tokens\n", model.cfg.n_layer, model.cfg.n_head, model.cfg.d_model, step, windows, ctx);
    println!("{:<8}{:>8}{:>10}{:>8}{:>9}{:>10}", "head", "self", "previous", "first", "other", "spread");
    let stride = (ids.len() - ctx) / windows.max(1);
    let (nl, nh) = (model.cfg.n_layer, model.cfg.n_head);
    let mut acc = vec![[0.0f64; 5]; nl * nh]; // self, previous, first, other, spread
    let mut count = 0.0f64;
    for w in 0..windows {
        let s = w * stride;
        let trace = model.attention_trace(&ids[s..s + ctx]);
        for l in 0..nl {
            for h in 0..nh {
                let m = &trace.weights[l][h];
                // skip the first few positions: with 1-2 visible tokens every pattern looks the same
                for i in 4..ctx {
                    let row: Vec<f32> = (0..=i).map(|j| m.get(i, j)).collect();
                    let a = &mut acc[l * nh + h];
                    a[0] += row[i] as f64;
                    a[1] += row[i - 1] as f64;
                    a[2] += row[0] as f64;
                    a[3] += (1.0 - row[i] - row[i - 1] - row[0]).max(0.0) as f64;
                    let entropy: f32 = row.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
                    a[4] += (entropy / ((i + 1) as f32).ln()) as f64;
                    if l == 0 && h == 0 {
                        count += 1.0;
                    }
                }
            }
        }
    }
    for l in 0..nl {
        for h in 0..nh {
            let a = acc[l * nh + h];
            println!("L{}H{:<5}{:>8.2}{:>10.2}{:>8.2}{:>9.2}{:>10.2}", l + 1, h + 1, a[0] / count, a[1] / count, a[2] / count, a[3] / count, a[4] / count);
        }
    }
}
