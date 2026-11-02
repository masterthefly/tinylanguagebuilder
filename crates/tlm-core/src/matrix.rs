/// A 2D grid of f32 numbers.
///
/// Memory layout is "row-major": the numbers are stored in ONE flat Vec,
/// row after row. For a 2x3 matrix
///
/// ```text
/// [ a b c ]
/// [ d e f ]
/// ```
///
/// the Vec holds [a, b, c, d, e, f]. The element at (row r, col c) lives at
/// index `r * cols + c`. Flat memory is what CPUs like best.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    /// A matrix full of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// Build a matrix from existing numbers (row-major order).
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "data length must equal rows * cols");
        Matrix { rows, cols, data }
    }

    /// Read the element at (r, c).
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    /// Write the element at (r, c).
    pub fn set(&mut self, r: usize, c: usize, value: f32) {
        self.data[r * self.cols + c] = value;
    }

    /// Element-wise addition: both matrices must have the same shape.
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols), "shape mismatch in add");
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }

    /// Swap rows and columns: element (r, c) moves to (c, r).
    pub fn transpose(&self) -> Matrix {
        let _t = crate::profile::span("transpose");
        let mut out = Matrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    /// Matrix multiplication: (m x k) times (k x n) gives (m x n).
    ///
    /// out[i][j] = sum over p of  self[i][p] * other[p][j]
    ///
    /// Uses the register-blocked kernel below (and AVX2+FMA when the CPU has them).
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "inner dimensions must match in matmul");
        #[cfg(feature = "naive-matmul")]
        return self.matmul_naive(other);
        let (m, k, n) = (self.rows, self.cols, other.cols);
        let mut out = Matrix::zeros(m, n);
        let _t = crate::profile::span("matmul");
        let left = Left { data: &self.data, row_stride: k, col_stride: 1 };
        matmul_into(left, &other.data, &mut out.data, m, k, n);
        out
    }

    /// `selfᵀ * other` without building the transpose.
    /// self is (k x m), other is (k x n), result is (m x n).
    pub fn matmul_at_b(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "row counts must match in matmul_at_b");
        #[cfg(feature = "naive-matmul")]
        return self.transpose().matmul_naive(other);
        let (k, m, n) = (self.rows, self.cols, other.cols);
        let mut out = Matrix::zeros(m, n);
        let _t = crate::profile::span("matmul");
        let left = Left { data: &self.data, row_stride: 1, col_stride: m };
        matmul_into(left, &other.data, &mut out.data, m, k, n);
        out
    }

    /// The simple version: kept as a speed baseline and a reference for tests.
    ///
    /// The loop order i, p, j is deliberate: the innermost loop walks along a
    /// row of `other` and a row of `out`, which sit next to each other in
    /// memory. That makes it much faster than the textbook i, j, p order.
    pub fn matmul_naive(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "inner dimensions must match in matmul");
        let (m, k, n) = (self.rows, self.cols, other.cols);
        let mut out = Matrix::zeros(m, n);
        for i in 0..m {
            for p in 0..k {
                let a = self.data[i * k + p];
                for j in 0..n {
                    out.data[i * n + j] += a * other.data[p * n + j];
                }
            }
        }
        out
    }
}

// ---- fast matrix multiply ------------------------------------------------------

/// Tile of the output kept in registers: MR rows by NR columns.
const MR: usize = 4;
const NR: usize = 16;

/// How the left matrix is laid out in memory. Element (i, p) of the matrix
/// being multiplied lives at `data[i * row_stride + p * col_stride]`.
///   normal matrix (m x k):                row_stride = k, col_stride = 1
///   transpose of a stored (k x m) matrix: row_stride = 1, col_stride = m
/// Reading with swapped strides means we never have to build the transpose.
#[derive(Clone, Copy)]
struct Left<'a> {
    data: &'a [f32],
    row_stride: usize,
    col_stride: usize,
}

/// out (m x n, zero-initialised) = left (m x k) * b (k x n); `b` and `out` row-major.
fn matmul_into(left: Left, b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert!(b.len() == k * n && out.len() == m * n);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: we just checked that this CPU supports AVX2 and FMA.
            unsafe { matmul_avx2_fma(left, b, out, m, k, n) };
            return;
        }
    }
    matmul_kernel::<false>(left, b, out, m, k, n);
}

/// The same kernel, compiled with AVX2 + FMA instructions switched on.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_avx2_fma(left: Left, b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    matmul_kernel::<true>(left, b, out, m, k, n);
}

/// `FMA` selects fused multiply-add. It must only be true when the hardware has it:
/// without hardware support, `mul_add` falls back to a very slow software routine.
#[inline(always)]
fn matmul_kernel<const FMA: bool>(left: Left, b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    let mut i0 = 0;
    while i0 + MR <= m {
        row_block::<MR, FMA>(left, b, out, i0, k, n);
        i0 += MR;
    }
    while i0 < m {
        row_block::<1, FMA>(left, b, out, i0, k, n);
        i0 += 1;
    }
}

/// Compute rows i0..i0+R of the output, one NR-wide tile at a time.
#[inline(always)]
fn row_block<const R: usize, const FMA: bool>(left: Left, b: &[f32], out: &mut [f32], i0: usize, k: usize, n: usize) {
    let a_at = |r: usize, p: usize| left.data[(i0 + r) * left.row_stride + p * left.col_stride];
    let n_full = n / NR * NR;

    for j0 in (0..n_full).step_by(NR) {
        // The tile lives in registers for the whole k loop.
        let mut acc = [[0.0f32; NR]; R];
        for p in 0..k {
            let b_tile: &[f32; NR] = b[p * n + j0..p * n + j0 + NR].try_into().unwrap();
            for r in 0..R {
                let av = a_at(r, p);
                for c in 0..NR {
                    acc[r][c] = if FMA { av.mul_add(b_tile[c], acc[r][c]) } else { acc[r][c] + av * b_tile[c] };
                }
            }
        }
        for r in 0..R {
            out[(i0 + r) * n + j0..(i0 + r) * n + j0 + NR].copy_from_slice(&acc[r]);
        }
    }

    // Leftover columns (when n is not a multiple of NR): plain loop.
    if n_full < n {
        for r in 0..R {
            for p in 0..k {
                let av = a_at(r, p);
                for j in n_full..n {
                    out[(i0 + r) * n + j] += av * b[p * n + j];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_matmul_matches_naive_on_awkward_shapes() {
        // Shapes chosen to hit every edge: fewer rows than a tile, rows not a
        // multiple of 4, columns not a multiple of 16, k = 1, 1x1.
        let mut rng = crate::Rng::new(9);
        let shapes = [(1, 1, 1), (1, 7, 1), (3, 5, 2), (4, 4, 16), (5, 3, 17), (9, 64, 33), (32, 64, 512), (7, 130, 50)];
        for (m, k, n) in shapes {
            let a = Matrix::from_vec(m, k, (0..m * k).map(|_| rng.next_f32() - 0.5).collect());
            let b = Matrix::from_vec(k, n, (0..k * n).map(|_| rng.next_f32() - 0.5).collect());
            let (fast, slow) = (a.matmul(&b), a.matmul_naive(&b));
            for (x, y) in fast.data.iter().zip(&slow.data) {
                assert!((x - y).abs() < 1e-4, "shape {m}x{k}x{n}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn matmul_at_b_equals_transpose_then_multiply() {
        let mut rng = crate::Rng::new(6);
        for (k, m, n) in [(1, 1, 1), (5, 3, 2), (32, 64, 17), (7, 9, 40), (64, 5, 64)] {
            let a = Matrix::from_vec(k, m, (0..k * m).map(|_| rng.next_f32() - 0.5).collect());
            let b = Matrix::from_vec(k, n, (0..k * n).map(|_| rng.next_f32() - 0.5).collect());
            let (fast, slow) = (a.matmul_at_b(&b), a.transpose().matmul_naive(&b));
            assert_eq!((fast.rows, fast.cols), (m, n));
            for (x, y) in fast.data.iter().zip(&slow.data) {
                assert!((x - y).abs() < 1e-4, "shape k{k} m{m} n{n}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn baseline_and_fma_kernels_agree() {
        // Both compiled variants must give (nearly) the same answer, so the
        // result does not depend on which one the CPU happens to select.
        let mut rng = crate::Rng::new(2);
        let (m, k, n) = (10, 40, 37);
        let a: Vec<f32> = (0..m * k).map(|_| rng.next_f32() - 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.next_f32() - 0.5).collect();
        let left = Left { data: &a, row_stride: k, col_stride: 1 };
        let mut plain = vec![0.0; m * n];
        matmul_kernel::<false>(left, &b, &mut plain, m, k, n);
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut fused = vec![0.0; m * n];
            // SAFETY: feature support was just checked.
            unsafe { matmul_avx2_fma(left, &b, &mut fused, m, k, n) };
            for (x, y) in plain.iter().zip(&fused) {
                assert!((x - y).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn get_and_set() {
        let mut m = Matrix::zeros(2, 3);
        m.set(1, 2, 7.0);
        assert_eq!(m.get(1, 2), 7.0);
        assert_eq!(m.data[5], 7.0); // row 1 * 3 cols + col 2 = index 5
    }

    #[test]
    fn add_works() {
        let a = Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let b = Matrix::from_vec(1, 3, vec![10.0, 20.0, 30.0]);
        assert_eq!(a.add(&b).data, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn transpose_works() {
        // [1 2 3]      [1 4]
        // [4 5 6]  ->  [2 5]
        //              [3 6]
        let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        assert_eq!((t.rows, t.cols), (3, 2));
        assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn matmul_works() {
        // [1 2]   [5 6]   [1*5+2*7  1*6+2*8]   [19 22]
        // [3 4] x [7 8] = [3*5+4*7  3*6+4*8] = [43 50]
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(a.matmul(&b).data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_non_square() {
        // (1x3) x (3x2) = (1x2)
        let a = Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let b = Matrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let c = a.matmul(&b);
        assert_eq!((c.rows, c.cols), (1, 2));
        assert_eq!(c.data, vec![4.0, 5.0]);
    }
}
