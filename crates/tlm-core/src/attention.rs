//! Single-head causal self-attention, forward and backward, by hand.
//!
//!   Q = X Wq    K = X Wk    V = X Wv
//!   S = Q Kᵀ / sqrt(d)            (T x T scores: token i vs token j)
//!   A = causal_softmax(S)         (row i only looks at tokens j <= i)
//!   O = A V                       (T x d output)
//!
//! X is one sequence: T rows (tokens) by d_model columns.

use crate::mlp::random_matrix;
use crate::ops::softmax;
use crate::{Matrix, Rng};

pub struct Attention {
    pub d_model: usize,
    pub d_head: usize,
    pub wq: Matrix, // d_model x d_head
    pub wk: Matrix,
    pub wv: Matrix,
}

/// Values from the forward pass that backward needs again.
pub struct AttnCache {
    pub x: Matrix,
    pub q: Matrix,
    pub k: Matrix,
    pub v: Matrix,
    /// The attention weights (T x T, lower-triangular, rows sum to 1).
    pub a: Matrix,
}

pub struct AttnGrads {
    pub dwq: Matrix,
    pub dwk: Matrix,
    pub dwv: Matrix,
    /// Gradient flowing back to the input, for the layer below.
    pub dx: Matrix,
}

/// Softmax over each row, but only across columns 0..=row (the past and present).
/// Everything to the right of the diagonal stays exactly 0.
pub fn causal_softmax_rows(s: &Matrix) -> Matrix {
    let t = s.rows;
    let mut out = Matrix::zeros(t, s.cols);
    for i in 0..t {
        let probs = softmax(&s.data[i * s.cols..i * s.cols + i + 1]);
        out.data[i * s.cols..i * s.cols + i + 1].copy_from_slice(&probs);
    }
    out
}

fn scale(m: &mut Matrix, factor: f32) {
    for v in m.data.iter_mut() {
        *v *= factor;
    }
}

impl Attention {
    pub fn new(d_model: usize, d_head: usize, rng: &mut Rng) -> Self {
        Attention {
            d_model,
            d_head,
            wq: random_matrix(d_model, d_head, d_model, rng),
            wk: random_matrix(d_model, d_head, d_model, rng),
            wv: random_matrix(d_model, d_head, d_model, rng),
        }
    }

    pub fn forward(&self, x: &Matrix) -> (Matrix, AttnCache) {
        assert_eq!(x.cols, self.d_model);
        let q = x.matmul(&self.wq);
        let k = x.matmul(&self.wk);
        let v = x.matmul(&self.wv);

        let mut s = q.matmul(&k.transpose()); // T x T
        scale(&mut s, 1.0 / (self.d_head as f32).sqrt());
        let a = causal_softmax_rows(&s);

        let out = a.matmul(&v);
        (out, AttnCache { x: x.clone(), q, k, v, a })
    }

    /// Given dL/dO, return dL/dWq, dL/dWk, dL/dWv and dL/dX.
    pub fn backward(&self, c: &AttnCache, dout: &Matrix) -> AttnGrads {
        let t = c.a.rows;
        let inv_sqrt_d = 1.0 / (self.d_head as f32).sqrt();

        // O = A V
        let da = dout.matmul(&c.v.transpose()); // T x T
        let dv = c.a.transpose().matmul(dout); // T x d

        // Softmax backward, row by row:  dS = A * (dA - sum_j(A * dA)).
        // Masked entries have A = 0, so their dS is automatically 0.
        let mut ds = Matrix::zeros(t, t);
        for i in 0..t {
            let row = i * t..(i + 1) * t;
            let dot: f32 = c.a.data[row.clone()].iter().zip(&da.data[row.clone()]).map(|(a, g)| a * g).sum();
            for j in 0..t {
                ds.data[i * t + j] = c.a.data[i * t + j] * (da.data[i * t + j] - dot);
            }
        }

        // S = Q Kᵀ / sqrt(d)
        let mut dq = ds.matmul(&c.k);
        scale(&mut dq, inv_sqrt_d);
        let mut dk = ds.transpose().matmul(&c.q);
        scale(&mut dk, inv_sqrt_d);

        // Q = X Wq, etc.
        let xt = c.x.transpose();
        let dwq = xt.matmul(&dq);
        let dwk = xt.matmul(&dk);
        let dwv = xt.matmul(&dv);

        // X feeds all three projections, so its gradient is the SUM of three paths.
        let dx = dq
            .matmul(&self.wq.transpose())
            .add(&dk.matmul(&self.wk.transpose()))
            .add(&dv.matmul(&self.wv.transpose()));

        AttnGrads { dwq, dwk, dwv, dx }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_input(rows: usize, cols: usize, rng: &mut Rng) -> Matrix {
        let data = (0..rows * cols).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        Matrix::from_vec(rows, cols, data)
    }

    #[test]
    fn weights_are_causal_and_sum_to_one() {
        let mut rng = Rng::new(1);
        let attn = Attention::new(6, 4, &mut rng);
        let x = random_input(5, 6, &mut rng);
        let (_, cache) = attn.forward(&x);
        for i in 0..5 {
            let row_sum: f32 = (0..5).map(|j| cache.a.get(i, j)).sum();
            assert!((row_sum - 1.0).abs() < 1e-5);
            for j in (i + 1)..5 {
                assert_eq!(cache.a.get(i, j), 0.0, "token {i} must not see future token {j}");
            }
        }
        // The first token can only see itself.
        assert!((cache.a.get(0, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn changing_the_future_does_not_change_the_past() {
        let mut rng = Rng::new(2);
        let attn = Attention::new(6, 4, &mut rng);
        let x1 = random_input(5, 6, &mut rng);
        let mut x2 = x1.clone();
        for c in 0..6 {
            x2.set(4, c, 99.0); // rewrite only the LAST token
        }
        let (o1, _) = attn.forward(&x1);
        let (o2, _) = attn.forward(&x2);
        assert_eq!(&o1.data[..4 * 4], &o2.data[..4 * 4]); // first 4 rows identical
        assert_ne!(&o1.data[4 * 4..], &o2.data[4 * 4..]); // last row changed
    }

    #[test]
    fn all_gradients_match_numerical_estimate() {
        let mut rng = Rng::new(3);
        let mut attn = Attention::new(4, 3, &mut rng);
        let mut x = random_input(4, 4, &mut rng);
        // Pretend the rest of the network is "multiply output by fixed R and sum".
        // Then dL/dO is just R, and L is a plain number we can nudge.
        let r = random_input(4, 3, &mut rng);

        let loss = |attn: &Attention, x: &Matrix| -> f32 {
            let (o, _) = attn.forward(x);
            o.data.iter().zip(&r.data).map(|(a, b)| a * b).sum()
        };

        let (_, cache) = attn.forward(&x);
        let grads = attn.backward(&cache, &r);
        let h = 1e-2;

        // Pick which matrix to nudge: 0 = wq, 1 = wk, 2 = wv, 3 = the input x.
        fn target<'a>(attn: &'a mut Attention, x: &'a mut Matrix, which: usize) -> &'a mut Matrix {
            match which {
                0 => &mut attn.wq,
                1 => &mut attn.wk,
                2 => &mut attn.wv,
                _ => x,
            }
        }

        let analytic_grads = [("wq", &grads.dwq), ("wk", &grads.dwk), ("wv", &grads.dwv), ("x", &grads.dx)];
        for (which, (name, analytic)) in analytic_grads.iter().enumerate() {
            for idx in 0..analytic.data.len() {
                let original = target(&mut attn, &mut x, which).data[idx];
                target(&mut attn, &mut x, which).data[idx] = original + h;
                let up = loss(&attn, &x);
                target(&mut attn, &mut x, which).data[idx] = original - h;
                let down = loss(&attn, &x);
                target(&mut attn, &mut x, which).data[idx] = original;

                let numeric = (up - down) / (2.0 * h);
                assert!(
                    (numeric - analytic.data[idx]).abs() < 3e-3,
                    "{name}[{idx}]: numeric {numeric} vs analytic {}",
                    analytic.data[idx]
                );
            }
        }
    }
}
