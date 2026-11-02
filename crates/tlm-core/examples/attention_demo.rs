//! See attention working, with hand-picked weights (no training yet).
//! Run with:  cargo run --release --example attention_demo
//!
//! Setup: 4 token types A,B,C,D as one-hot vectors. We set Wq = Wk = 3 * identity,
//! so a query and key score high exactly when they are the SAME token type.
//! Result: each token pays attention to earlier copies of itself.

use tlm_core::attention::Attention;
use tlm_core::{Matrix, Rng};

fn main() {
    let names = ['A', 'B', 'C', 'D'];
    let sequence = [0, 1, 0, 2, 0, 1]; // A B A C A B

    let mut x = Matrix::zeros(sequence.len(), 4);
    for (row, &tok) in sequence.iter().enumerate() {
        x.set(row, tok, 1.0);
    }

    let mut rng = Rng::new(1);
    let mut attn = Attention::new(4, 4, &mut rng);
    for i in 0..4 {
        for j in 0..4 {
            let on_diag = if i == j { 3.0 } else { 0.0 };
            attn.wq.set(i, j, on_diag);
            attn.wk.set(i, j, on_diag);
            attn.wv.set(i, j, if i == j { 1.0 } else { 0.0 });
        }
    }

    let (out, cache) = attn.forward(&x);

    let seq_str: String = sequence.iter().map(|&t| names[t]).collect();
    println!("sequence: {seq_str}\n");
    println!("attention weights (row = who is looking, column = who they look at):\n");
    print!("      ");
    for &t in &sequence {
        print!("   {}  ", names[t]);
    }
    println!();
    for i in 0..sequence.len() {
        print!("  {}   ", names[sequence[i]]);
        for j in 0..sequence.len() {
            if j > i {
                print!("   .  "); // masked: the future
            } else {
                print!(" {:.2} ", cache.a.get(i, j));
            }
        }
        println!();
    }

    println!("\noutput for the last token (a blend of the Value vectors it attended to):");
    println!("  A={:.2}  B={:.2}  C={:.2}  D={:.2}", out.get(5, 0), out.get(5, 1), out.get(5, 2), out.get(5, 3));
    println!("The final B attends mostly to the earlier B, so its output is mostly 'B'.");
}
