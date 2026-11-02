//! Measure matrix-multiply speed and a full training step.
//! Run with:  cargo run --release --example bench

use std::time::Instant;
use tlm_core::transformer::{Transformer, TransformerConfig};
use tlm_core::{Matrix, Rng};

fn random(rows: usize, cols: usize, rng: &mut Rng) -> Matrix {
    Matrix::from_vec(rows, cols, (0..rows * cols).map(|_| rng.next_f32() - 0.5).collect())
}

/// Run `f` repeatedly for about `budget` seconds; return the FASTEST single call.
/// Other programs, power saving and thermal limits can only ever make a run
/// slower, never faster, so the minimum is the most repeatable number.
fn time_it(budget: f32, mut f: impl FnMut()) -> f32 {
    f(); // warm up caches
    let start = Instant::now();
    let mut best = f32::MAX;
    while start.elapsed().as_secs_f32() < budget {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f32());
    }
    best
}

fn main() {
    let mut rng = Rng::new(1);

    // Shapes from our default model (d_model 64, 512 tokens per step, vocab 512).
    let shapes = [
        ("project (T x d)(d x d)", 32, 64, 64),
        ("ffn expand (T x d)(d x 4d)", 32, 64, 256),
        ("ffn shrink (T x 4d)(4d x d)", 32, 256, 64),
        ("output (T x d)(d x vocab)", 32, 64, 512),
        ("big square 256", 256, 256, 256),
    ];
    println!("{:<30} {:>12} {:>12} {:>9}", "matmul shape", "naive GFLOPS", "fast GFLOPS", "speedup");
    for (name, m, k, n) in shapes {
        let (a, b) = (random(m, k, &mut rng), random(k, n, &mut rng));
        let flops = 2.0 * (m * k * n) as f32;
        let naive = time_it(0.4, || {
            std::hint::black_box(a.matmul_naive(&b));
        });
        let fast = time_it(0.4, || {
            std::hint::black_box(a.matmul(&b));
        });
        println!(
            "{name:<30} {:>12.1} {:>12.1} {:>8.1}x",
            flops / naive / 1e9,
            flops / fast / 1e9,
            naive / fast
        );
    }

    // End-to-end: one training step on 16 sequences of 32 tokens, ONE thread.
    let cfg = TransformerConfig { vocab_size: 512, ctx: 32, d_model: 64, n_layer: 2, n_head: 4 };
    let model = Transformer::new(cfg, &mut rng);
    let ids: Vec<usize> = (0..16 * 32).map(|i| (i * 7) % 512).collect();
    let tgt: Vec<usize> = ids.iter().map(|i| (i + 1) % 512).collect();
    tlm_core::profile::take(); // discard anything recorded so far
    let started = Instant::now();
    let mut steps = 0;
    let secs = time_it(1.5, || {
        std::hint::black_box(model.loss_and_grads(&ids, &tgt, 32, 1));
        steps += 1;
    });
    let total = started.elapsed().as_secs_f64();
    println!("\ntraining step (1 thread, 512 tokens): {:.2} ms  ->  {:.0} tokens/s", secs * 1e3, 512.0 / secs);

    if tlm_core::profile::ENABLED {
        println!("\nwhere the time goes (inclusive; nested spans overlap):");
        println!("{:<22} {:>10} {:>12} {:>8}", "span", "calls/step", "ms/step", "share");
        for (name, calls, secs) in tlm_core::profile::take().into_iter().take(14) {
            println!(
                "{name:<22} {:>10.0} {:>12.2} {:>7.0}%",
                calls as f64 / steps as f64,
                1e3 * secs / steps as f64,
                100.0 * secs / total
            );
        }
        println!("(profiling itself slows the step down a little)");
    } else {
        println!("(run with --features profile to see where the time goes)");
    }
}
