//! Where does the time go OUTSIDE matmul? Time the building blocks and scale
//! each by how many times one training step uses it.
//! Run with:  cargo run --release --example bench_ops

use std::time::Instant;
use tlm_core::ops::softmax_rows;
use tlm_core::{Matrix, Rng};

fn time_it(mut f: impl FnMut()) -> f32 {
    f();
    let start = Instant::now();
    let mut runs = 0;
    while start.elapsed().as_secs_f32() < 0.3 {
        f();
        runs += 1;
    }
    start.elapsed().as_secs_f32() / runs as f32
}

fn main() {
    let mut rng = Rng::new(1);
    let hidden: Vec<f32> = (0..32 * 256).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
    let logits = Matrix::from_vec(32, 512, (0..32 * 512).map(|_| rng.next_f32() * 4.0 - 2.0).collect());
    let act = Matrix::from_vec(32, 256, hidden.clone());

    // One training step = 16 sequences. Per sequence: 2 layers.
    let tanh_t = time_it(|| {
        let mut s = 0.0;
        for &x in &hidden {
            s += std::hint::black_box(x).tanh();
        }
        std::hint::black_box(s);
    });
    let softmax_t = time_it(|| {
        std::hint::black_box(softmax_rows(&logits));
    });
    let transpose_t = time_it(|| {
        std::hint::black_box(act.transpose());
    });
    let alloc_t = time_it(|| {
        std::hint::black_box(Matrix::zeros(32, 256));
    });

    println!("{:<44} {:>9} {:>9} {:>11}", "operation", "per call", "x per step", "ms per step");
    let rows = [
        // gelu forward + gelu gradient each call tanh on 32x256, in 2 layers, 16 sequences
        ("tanh on 32x256 (GELU fwd + grad)", tanh_t, 2.0 * 2.0 * 16.0),
        // loss softmax over 32x512 once per sequence
        ("softmax_rows 32x512 (loss)", softmax_t, 16.0),
        // backward transposes a handful of 32x256-ish matrices per layer
        ("transpose 32x256 (~8 per layer)", transpose_t, 8.0 * 2.0 * 16.0),
        ("allocate 32x256 (~100 per sequence)", alloc_t, 100.0 * 16.0),
    ];
    let mut total = 0.0;
    for (name, per_call, count) in rows {
        println!("{name:<44} {:>7.1}us {:>9.0} {:>10.2}", per_call * 1e6, count, per_call * count * 1e3);
        total += per_call * count * 1e3;
    }
    println!("\nestimated non-matmul cost from these: {total:.1} ms of a ~20 ms step");
}
