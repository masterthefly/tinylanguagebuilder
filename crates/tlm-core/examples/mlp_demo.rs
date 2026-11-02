//! Train the context-aware MLP and compare it with the bigram baseline.
//! Run with:  cargo run --release --example mlp_demo

use tlm_core::bigram::{Bigram, CharVocab};
use tlm_core::mlp::{make_dataset, Mlp, MlpConfig};
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

    // Hold out the last 20% to measure how well the model handles text it
    // has NOT trained on. Training loss alone can lie (memorising).
    let split = ids.len() * 8 / 10;
    let (train_ids, val_ids) = (&ids[..split], &ids[split..]);

    // Baseline: the bigram from last lesson.
    let mut bigram = Bigram::new(vocab.size());
    for _ in 0..300 {
        bigram.train_step(&train_ids[..train_ids.len() - 1], &train_ids[1..], 5.0);
    }
    let (bigram_val, _) = bigram.loss_and_grad(&val_ids[..val_ids.len() - 1], &val_ids[1..]);
    println!("bigram baseline      val loss {bigram_val:.3}\n");

    let cfg = MlpConfig { vocab_size: vocab.size(), context: 4, emb_dim: 8, hidden: 64 };
    let mut rng = Rng::new(1234);
    let mut model = Mlp::new(cfg, &mut rng);
    println!("MLP parameters: {}", model.num_params());

    let (train_ctx, train_t) = make_dataset(train_ids, cfg.context);
    let (val_ctx, val_t) = make_dataset(val_ids, cfg.context);

    for step in 0..=1500 {
        let train_loss = model.train_step(&train_ctx, &train_t, 0.3);
        if step % 250 == 0 {
            let val_loss = model.loss(&val_ctx, &val_t);
            println!("step {step:>4}  train loss {train_loss:.3}   val loss {val_loss:.3}");
        }
    }

    println!("\n--- generated text ---");
    let start = vocab.encode("the ");
    println!("the {}", vocab.decode(&model.sample(&start, 300, &mut rng)));
}
