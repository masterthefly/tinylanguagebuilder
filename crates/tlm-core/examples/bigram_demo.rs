//! Train a bigram model on a small piece of text and generate from it.
//! Run with:  cargo run --release --example bigram_demo

use tlm_core::bigram::{Bigram, CharVocab};
use tlm_core::Rng;

const TEXT: &str = "\
the little robot lived in a small house on a hill. every morning the robot \
walked down to the river to look at the water. the water was cold and clear. \
one day the robot found a red boat by the river. the robot wanted to sail \
the boat to the sea. the sea was far away but the robot was not afraid. \
it packed a lunch and a lamp and pushed the boat into the water. the river \
carried the boat past the green hills and the tall trees. the robot sang \
a little song as the boat floated on and on toward the sea.";

fn main() {
    let vocab = CharVocab::from_text(TEXT);
    let ids = vocab.encode(TEXT);
    let (inputs, targets) = (&ids[..ids.len() - 1], &ids[1..]);

    println!("characters in text: {}", ids.len());
    println!("vocabulary size:    {}", vocab.size());
    println!("untrained loss would be ln({}) = {:.3}\n", vocab.size(), (vocab.size() as f32).ln());

    let mut model = Bigram::new(vocab.size());
    for step in 0..=300 {
        let loss = model.train_step(inputs, targets, 5.0);
        if step % 50 == 0 {
            println!("step {step:>3}  loss {loss:.3}");
        }
    }

    println!("\n--- generated text ---");
    let mut rng = Rng::new(2024);
    let start = vocab.encode("t")[0];
    println!("{}", vocab.decode(&model.sample(start, 300, &mut rng)));
}
