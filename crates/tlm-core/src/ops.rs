//! Math operations used by the model (beyond plain Matrix arithmetic).

use crate::Matrix;

/// The textbook softmax. It is mathematically correct but BREAKS on large
/// inputs. It exists only so we can watch it fail in a test. Never use it.
pub fn softmax_naive(x: &[f32]) -> Vec<f32> {
    let exps: Vec<f32> = x.iter().map(|v| v.exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Numerically stable softmax.
///
/// Subtracting the maximum first does not change the result (the factor
/// cancels in the division) but guarantees the largest exp() is exp(0) = 1,
/// so nothing can overflow.
pub fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Pick an index at random, weighted by `probs` (which must sum to ~1).
/// Roll a die in [0,1) and walk along the cumulative sum until we pass it.
pub fn sample_from_probs(probs: &[f32], rng: &mut crate::Rng) -> usize {
    let roll = rng.next_f32();
    let mut cumulative = 0.0;
    for (j, &p) in probs.iter().enumerate() {
        cumulative += p;
        if roll < cumulative {
            return j;
        }
    }
    probs.len() - 1 // fallback for floating-point rounding
}

/// Apply softmax to every row of a matrix independently.
/// Attention will use this: each row becomes "how much to look at each token".
pub fn softmax_rows(m: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(m.rows, m.cols);
    for r in 0..m.rows {
        let row = &m.data[r * m.cols..(r + 1) * m.cols];
        let probs = softmax(row);
        out.data[r * m.cols..(r + 1) * m.cols].copy_from_slice(&probs);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn softmax_known_values() {
        let p = softmax(&[2.0, 1.0, 0.1]);
        assert!(close(p[0], 0.65900114));
        assert!(close(p[1], 0.24243297));
        assert!(close(p[2], 0.09856589));
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[3.0, -1.0, 0.5, 2.2]);
        assert!(close(p.iter().sum::<f32>(), 1.0));
    }

    #[test]
    fn naive_softmax_breaks_on_large_input() {
        // exp(1000) overflows to infinity, and inf / inf is NaN.
        let p = softmax_naive(&[1000.0, 1001.0]);
        assert!(p[0].is_nan());
    }

    #[test]
    fn stable_softmax_survives_large_input() {
        let p = softmax(&[1000.0, 1001.0]);
        assert!(close(p[0], 0.26894143));
        assert!(close(p[1], 0.7310586));
    }

    #[test]
    fn softmax_ignores_constant_shift() {
        let a = softmax(&[1.0, 2.0, 3.0]);
        let b = softmax(&[101.0, 102.0, 103.0]);
        for i in 0..3 {
            assert!(close(a[i], b[i]));
        }
    }

    #[test]
    fn softmax_rows_each_row_sums_to_one() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 10.0, 0.0, -10.0]);
        let s = softmax_rows(&m);
        for r in 0..2 {
            let total: f32 = (0..3).map(|c| s.get(r, c)).sum();
            assert!(close(total, 1.0));
        }
    }
}
