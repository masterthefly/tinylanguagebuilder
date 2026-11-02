//! Optimizer: AdamW, plus gradient clipping.
//!
//! Plain gradient descent uses ONE learning rate for every weight. Adam keeps,
//! for each weight, a running average of its gradient (m, "momentum") and of
//! its squared gradient (v). Weights with consistently large gradients get
//! smaller steps; rarely-updated weights get larger ones.
//!
//!   m = b1*m + (1-b1)*g          v = b2*v + (1-b2)*g^2
//!   w -= lr * m_hat / (sqrt(v_hat) + eps)         (hats = bias-corrected)
//!
//! "W" = weight decay applied directly to the weights (not through the
//! gradient), which gently pulls matrices toward zero and fights overfitting.

use crate::Matrix;

pub struct AdamW {
    pub lr: f32,
    pub weight_decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: i32,
    m: Vec<Matrix>,
    v: Vec<Matrix>,
}

impl AdamW {
    pub fn new(params: &[Matrix], lr: f32, weight_decay: f32) -> Self {
        let zeros = || params.iter().map(|p| Matrix::zeros(p.rows, p.cols)).collect();
        AdamW { lr, weight_decay, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0, m: zeros(), v: zeros() }
    }

    /// Number of updates applied so far.
    pub fn updates(&self) -> i32 {
        self.t
    }

    /// Adam's two running averages, for saving.
    pub fn moments(&self) -> (&[Matrix], &[Matrix]) {
        (&self.m, &self.v)
    }

    /// Put saved state back, after checking it matches this optimizer's shapes.
    pub fn restore(&mut self, t: i32, m: Vec<Matrix>, v: Vec<Matrix>) -> Result<(), String> {
        let same = |a: &[Matrix], b: &[Matrix]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x.rows, x.cols) == (y.rows, y.cols))
        };
        if !same(&m, &self.m) || !same(&v, &self.v) {
            return Err("saved optimizer state does not match the model's parameter shapes".into());
        }
        self.t = t;
        self.m = m;
        self.v = v;
        Ok(())
    }

    pub fn step(&mut self, params: &mut [Matrix], grads: &[Matrix]) {
        self.t += 1;
        let bias1 = 1.0 - self.beta1.powi(self.t);
        let bias2 = 1.0 - self.beta2.powi(self.t);
        for i in 0..params.len() {
            // Decay only real matrices; biases and norm gains (1 x N) are left alone.
            let decay = if params[i].rows > 1 { self.weight_decay } else { 0.0 };
            for k in 0..params[i].data.len() {
                let g = grads[i].data[k];
                let m = self.beta1 * self.m[i].data[k] + (1.0 - self.beta1) * g;
                let v = self.beta2 * self.v[i].data[k] + (1.0 - self.beta2) * g * g;
                self.m[i].data[k] = m;
                self.v[i].data[k] = v;
                let step = (m / bias1) / ((v / bias2).sqrt() + self.eps);
                let w = &mut params[i].data[k];
                *w -= self.lr * (step + decay * *w);
            }
        }
    }
}

/// If the overall gradient length exceeds `max_norm`, shrink it to exactly that.
/// Protects against one freak batch throwing the weights far off course.
/// Returns the norm measured before clipping.
pub fn clip_grad_norm(grads: &mut [Matrix], max_norm: f32) -> f32 {
    let sum_sq: f32 = grads.iter().flat_map(|g| g.data.iter()).map(|v| v * v).sum();
    let norm = sum_sq.sqrt();
    if norm > max_norm {
        let factor = max_norm / norm;
        for g in grads.iter_mut() {
            for v in g.data.iter_mut() {
                *v *= factor;
            }
        }
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_finds_the_minimum_of_a_parabola() {
        // Minimise (w - 3)^2, whose gradient is 2(w - 3).
        let mut params = vec![Matrix::zeros(1, 1)];
        let mut opt = AdamW::new(&params, 0.1, 0.0);
        for _ in 0..500 {
            let g = Matrix::from_vec(1, 1, vec![2.0 * (params[0].data[0] - 3.0)]);
            opt.step(&mut params, &[g]);
        }
        assert!((params[0].data[0] - 3.0).abs() < 0.05, "got {}", params[0].data[0]);
    }

    #[test]
    fn clipping_limits_the_norm() {
        let mut grads = vec![Matrix::from_vec(1, 2, vec![3.0, 4.0])]; // norm 5
        let before = clip_grad_norm(&mut grads, 1.0);
        assert!((before - 5.0).abs() < 1e-5);
        let after: f32 = grads[0].data.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((after - 1.0).abs() < 1e-5);
    }

    #[test]
    fn small_gradients_are_not_clipped() {
        let mut grads = vec![Matrix::from_vec(1, 2, vec![0.3, 0.4])];
        clip_grad_norm(&mut grads, 1.0);
        assert_eq!(grads[0].data, vec![0.3, 0.4]);
    }
}
