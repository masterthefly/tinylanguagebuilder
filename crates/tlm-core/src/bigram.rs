//! The simplest language model: look at ONE character, predict the next.
//! This is rung 1 of the model ladder. It learns with gradient descent.

use crate::ops::{sample_from_probs, softmax};
use crate::{Matrix, Rng};

/// Maps characters to integer ids and back. Models only understand numbers.
pub struct CharVocab {
    pub chars: Vec<char>,
}

impl CharVocab {
    /// Collect every distinct character in the text (sorted, so ids are stable).
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();
        CharVocab { chars }
    }

    pub fn size(&self) -> usize {
        self.chars.len()
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| self.chars.binary_search(&c).expect("character not in vocab"))
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&i| self.chars[i]).collect()
    }
}

/// Row `i` of `logits` holds the raw scores for "what comes after character i".
pub struct Bigram {
    pub vocab_size: usize,
    pub logits: Matrix,
}

impl Bigram {
    /// Starts knowing nothing: all zeros, so every next character is equally likely.
    pub fn new(vocab_size: usize) -> Self {
        Bigram { vocab_size, logits: Matrix::zeros(vocab_size, vocab_size) }
    }

    fn row(&self, i: usize) -> &[f32] {
        &self.logits.data[i * self.vocab_size..(i + 1) * self.vocab_size]
    }

    /// Average cross-entropy loss, plus the gradient of the loss with
    /// respect to every table entry.
    ///
    /// For each (input, target) pair:
    ///   loss     = -ln( probs[target] )
    ///   gradient = probs - one_hot(target)      (a famously simple result)
    pub fn loss_and_grad(&self, inputs: &[usize], targets: &[usize]) -> (f32, Matrix) {
        assert_eq!(inputs.len(), targets.len());
        let n = inputs.len() as f32;
        let mut grad = Matrix::zeros(self.vocab_size, self.vocab_size);
        let mut total_loss = 0.0;
        for (&x, &t) in inputs.iter().zip(targets) {
            let probs = softmax(self.row(x));
            total_loss += -probs[t].ln();
            for (j, &p) in probs.iter().enumerate() {
                let one_hot = if j == t { 1.0 } else { 0.0 };
                grad.data[x * self.vocab_size + j] += (p - one_hot) / n;
            }
        }
        (total_loss / n, grad)
    }

    /// One gradient-descent step: nudge every entry against its gradient.
    /// Returns the loss measured BEFORE the nudge.
    pub fn train_step(&mut self, inputs: &[usize], targets: &[usize], lr: f32) -> f32 {
        let (loss, grad) = self.loss_and_grad(inputs, targets);
        for (w, g) in self.logits.data.iter_mut().zip(&grad.data) {
            *w -= lr * g;
        }
        loss
    }

    /// Generate `len` new characters, starting after `start`.
    /// Each step: softmax the current row, then roll a weighted die.
    pub fn sample(&self, start: usize, len: usize, rng: &mut Rng) -> Vec<usize> {
        let mut out = Vec::with_capacity(len);
        let mut current = start;
        for _ in 0..len {
            let probs = softmax(self.row(current));
            let chosen = sample_from_probs(&probs, rng);
            out.push(chosen);
            current = chosen;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_round_trip() {
        let v = CharVocab::from_text("hello world");
        let ids = v.encode("hello");
        assert_eq!(v.decode(&ids), "hello");
    }

    #[test]
    fn untrained_loss_is_ln_vocab_size() {
        // Knowing nothing = guessing uniformly = loss of ln(V).
        let model = Bigram::new(27);
        let (loss, _) = model.loss_and_grad(&[0, 1, 2], &[3, 4, 5]);
        assert!((loss - (27.0f32).ln()).abs() < 1e-4);
    }

    #[test]
    fn gradient_matches_numerical_estimate() {
        // The golden rule: never trust a hand-written gradient until you
        // compare it with (loss(w+h) - loss(w-h)) / 2h.
        let mut model = Bigram::new(4);
        for (i, w) in model.logits.data.iter_mut().enumerate() {
            *w = (i as f32 * 0.37).sin(); // arbitrary non-trivial weights
        }
        let (inputs, targets) = (vec![0, 1, 2, 1, 3], vec![1, 2, 1, 3, 0]);
        let (_, grad) = model.loss_and_grad(&inputs, &targets);

        let h = 1e-2;
        for idx in 0..model.logits.data.len() {
            let original = model.logits.data[idx];
            model.logits.data[idx] = original + h;
            let (up, _) = model.loss_and_grad(&inputs, &targets);
            model.logits.data[idx] = original - h;
            let (down, _) = model.loss_and_grad(&inputs, &targets);
            model.logits.data[idx] = original;
            let numeric = (up - down) / (2.0 * h);
            assert!(
                (numeric - grad.data[idx]).abs() < 1e-3,
                "entry {idx}: numeric {numeric} vs analytic {}",
                grad.data[idx]
            );
        }
    }

    #[test]
    fn training_lowers_loss_and_learns_pattern() {
        // "abababab...": after 'a' always comes 'b', and vice versa.
        let text = "ab".repeat(50);
        let vocab = CharVocab::from_text(&text);
        let ids = vocab.encode(&text);
        let (inputs, targets) = (&ids[..ids.len() - 1], &ids[1..]);

        let mut model = Bigram::new(vocab.size());
        let first = model.train_step(inputs, targets, 1.0);
        let mut last = first;
        for _ in 0..200 {
            last = model.train_step(inputs, targets, 1.0);
        }
        assert!(last < first / 10.0, "loss should fall a lot: {first} -> {last}");

        // The model should now generate the alternating pattern.
        let mut rng = Rng::new(1);
        let sample = vocab.decode(&model.sample(vocab.encode("a")[0], 10, &mut rng));
        assert_eq!(sample, "babababab".to_string() + "a");
    }
}
