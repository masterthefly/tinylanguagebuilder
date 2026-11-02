//! A tiny deterministic random number generator (xorshift64*).
//! Same seed gives the same numbers on every machine, which makes
//! training runs and tests reproducible. No external crate needed.

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // State must never be zero for xorshift.
        Rng { state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed } }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// An integer in [0, n). (Tiny modulo bias; irrelevant for our purposes.)
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// A float in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // Use the top 24 bits: exactly what an f32 mantissa can hold.
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_numbers() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..10 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn floats_in_range() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let f = r.next_f32();
            assert!((0.0..1.0).contains(&f));
        }
    }
}
