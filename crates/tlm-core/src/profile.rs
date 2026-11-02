//! A tiny built-in profiler.
//!
//! Put `let _t = profile::span("name");` at the top of any function or block.
//! When the span is dropped, the elapsed time is added to the total for "name".
//! Without `--features profile` the span is empty and compiles to nothing.
//!
//! Spans can nest (a "matmul" span inside an "attn fwd" span), so the numbers
//! are INCLUSIVE times and the percentages may add up to more than 100%.

#[cfg(feature = "profile")]
mod imp {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    static TOTALS: Mutex<Option<HashMap<&'static str, (u64, u64)>>> = Mutex::new(None);

    pub struct Span {
        name: &'static str,
        start: Instant,
    }

    pub fn span(name: &'static str) -> Span {
        Span { name, start: Instant::now() }
    }

    impl Drop for Span {
        fn drop(&mut self) {
            let elapsed: Duration = self.start.elapsed();
            let mut guard = TOTALS.lock().unwrap();
            let entry = guard.get_or_insert_with(HashMap::new).entry(self.name).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += elapsed.as_nanos() as u64;
        }
    }

    pub fn take() -> Vec<(&'static str, u64, f64)> {
        let taken = TOTALS.lock().unwrap().take().unwrap_or_default();
        let mut rows: Vec<_> = taken.into_iter().map(|(name, (calls, nanos))| (name, calls, nanos as f64 / 1e9)).collect();
        rows.sort_by(|a, b| b.2.total_cmp(&a.2));
        rows
    }
}

#[cfg(not(feature = "profile"))]
mod imp {
    pub struct Span;
    #[inline(always)]
    pub fn span(_name: &'static str) -> Span {
        Span
    }
    pub fn take() -> Vec<(&'static str, u64, f64)> {
        Vec::new()
    }
}

pub use imp::{span, take, Span};

/// True when the crate was built with `--features profile`.
pub const ENABLED: bool = cfg!(feature = "profile");
