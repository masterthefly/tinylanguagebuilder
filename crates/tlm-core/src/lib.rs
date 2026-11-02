//! tlm-core: the math engine behind TinyLanguageModel.
//! No GPU, no external dependencies: just Rust and CPU.

pub mod attention;
pub mod bigram;
pub mod bpe;
pub mod checkpoint;
pub mod dpo;
pub mod matrix;
pub mod mlp;
pub mod ops;
pub mod optim;
pub mod parallel;
pub mod profile;
pub mod rng;
pub mod sft;
pub mod trainer;
pub mod transformer;

pub use matrix::Matrix;
pub use rng::Rng;
