//! Supervised fine-tuning (SFT): teach a base model to follow an instruction.
//!
//! Each example is (prompt, response). We train on `prompt + response` exactly
//! like pre-training, with ONE difference: the loss is only counted on the
//! response tokens. The prompt is context the model must read, not text it
//! should learn to write.
//!
//! ```text
//! tokens:   Word : ball \n Sentence :   A red ball rolled . <end>
//! learn?     no   no  no   no    no      YES YES  YES   YES  YES
//! ```

use crate::bpe::Tokenizer;
use crate::optim::{clip_grad_norm, AdamW};
use crate::parallel::shard_range;
use crate::trainer::lr_at;
use crate::transformer::{Transformer, IGNORE_TARGET};
use crate::{Matrix, Rng};
use serde_json::Value;

/// Marks the end of a response. The base data already uses it between stories.
pub const STOP: &str = "<|endoftext|>";

/// The prompt layout for the built-in "use the word" task. `{}` is the word.
pub const WORD_TEMPLATE: &str = "Word: {}\nSentence:";

#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    pub prompt: String,
    /// Must already end with `STOP` so the model learns when to finish.
    pub response: String,
}

/// An example turned into token ids: prompt tokens first, then response tokens.
#[derive(Debug, Clone)]
pub struct TokenizedExample {
    pub ids: Vec<usize>,
    pub prompt_len: usize,
}

/// Prompt and response are tokenised SEPARATELY and joined. At chat time the
/// prompt is also tokenised on its own, so training and use see identical boundaries.
pub fn tokenize(tok: &Tokenizer, ex: &Example) -> TokenizedExample {
    let mut ids = tok.encode(&ex.prompt);
    let prompt_len = ids.len();
    ids.extend(tok.encode(&ex.response));
    TokenizedExample { ids, prompt_len }
}

/// Lay the chosen examples out as one batch of fixed-length rows.
///
/// For each example the model reads `ids[..n-1]` and must predict `ids[1..]`.
/// The targets that fall inside the prompt are masked, and so is the padding
/// that fills each row up to `seq_len`. (Padding sits at the END, and attention
/// is causal, so it cannot influence the real tokens before it.)
///
/// Examples longer than `seq_len + 1` tokens must be filtered out beforehand.
pub fn pack(examples: &[TokenizedExample], chosen: &[usize], seq_len: usize) -> (Vec<usize>, Vec<usize>) {
    let mut inputs = vec![0usize; chosen.len() * seq_len];
    let mut targets = vec![IGNORE_TARGET; chosen.len() * seq_len];
    for (row, &i) in chosen.iter().enumerate() {
        let ex = &examples[i];
        let n = ex.ids.len() - 1; // number of (input, target) pairs
        assert!(n <= seq_len, "example too long for the context window");
        for t in 0..n {
            inputs[row * seq_len + t] = ex.ids[t];
            // Target t is token t+1: it belongs to the response iff t+1 >= prompt_len.
            if t + 1 >= ex.prompt_len {
                targets[row * seq_len + t] = ex.ids[t + 1];
            }
        }
    }
    (inputs, targets)
}

/// Keep only examples that fit in the context window, tokenised.
pub fn tokenize_all(tok: &Tokenizer, examples: &[Example], seq_len: usize) -> Vec<TokenizedExample> {
    examples
        .iter()
        .map(|e| tokenize(tok, e))
        .filter(|t| t.ids.len() >= 2 && t.ids.len() - 1 <= seq_len && t.prompt_len < t.ids.len())
        .collect()
}

// ---- building a dataset ------------------------------------------------------------

/// Words too common to make a good "use this word" prompt.
const STOPWORDS: &[&str] = &[
    "that", "with", "this", "they", "there", "their", "were", "what", "when", "have", "from", "then", "them",
    "said", "would", "could", "about", "into", "very", "just", "also", "been", "some", "than", "your", "will",
    "each", "after", "before", "because", "while", "where", "which", "other", "only", "over", "down", "again",
    "back", "like", "went", "came", "know", "want", "wanted", "looked", "named", "once", "upon", "time", "many",
    "more", "much", "most", "such", "even", "here", "these", "those", "does", "didn", "dont", "didnt", "cant",
];

/// The first sentence of `story`, if it is a reasonable size and plain text.
fn first_sentence(story: &str) -> Option<&str> {
    let bytes = story.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if matches!(b, b'.' | b'!' | b'?') && bytes.get(i + 1).is_none_or(|n| n.is_ascii_whitespace()) {
            let s = &story[..=i];
            let ok = (25..=140).contains(&s.len()) && s.is_ascii() && !s.contains('\n');
            return ok.then_some(s);
        }
    }
    None
}

/// Candidate topic words in a sentence: lowercase letters, 4-12 long, not a stopword.
fn topic_words(sentence: &str) -> Vec<String> {
    let mut seen = Vec::new();
    for w in sentence.split(|c: char| !c.is_ascii_alphabetic()) {
        let w = w.to_ascii_lowercase();
        if (4..=12).contains(&w.len()) && !STOPWORDS.contains(&w.as_str()) && !seen.contains(&w) {
            seen.push(w);
        }
    }
    seen
}

/// Build "use the word" examples from raw story text: the first sentence of each
/// story is the ideal answer, and one of its words is the instruction.
pub fn word_examples(text: &str, max: usize, rng: &mut Rng) -> Vec<Example> {
    let mut out = Vec::new();
    // Stories are separated by the stop marker; text without markers falls back to paragraphs.
    let separator = if text.contains(STOP) { STOP } else { "\n\n" };
    for story in text.split(separator) {
        if out.len() >= max {
            break;
        }
        let Some(sentence) = first_sentence(story.trim()) else { continue };
        let words = topic_words(sentence);
        if words.is_empty() {
            continue;
        }
        let word = &words[rng.next_usize(words.len())];
        out.push(Example { prompt: WORD_TEMPLATE.replace("{}", word), response: format!(" {sentence}{STOP}") });
    }
    out
}

/// Parse one example per line: {"prompt": "...", "response": "..."}.
/// `STOP` is appended to each response automatically.
pub fn parse_jsonl(text: &str) -> Result<Vec<Example>, String> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).map_err(|e| format!("line {}: {e}", n + 1))?;
        let field = |name: &str| v[name].as_str().map(str::to_string).ok_or(format!("line {}: missing string \"{name}\"", n + 1));
        out.push(Example { prompt: field("prompt")?, response: format!("{}{STOP}", field("response")?) });
    }
    if out.is_empty() {
        return Err("no examples found".into());
    }
    Ok(out)
}

// ---- training ------------------------------------------------------------------------

pub struct SftConfig {
    pub batch: usize,
    pub base_lr: f32,
    pub weight_decay: f32,
    pub warmup: usize,
    pub clip: f32,
    pub threads: usize,
}

pub struct SftTrainer {
    pub model: Transformer,
    pub opt: AdamW,
    pub cfg: SftConfig,
    pub step: usize,
    pub total_steps: usize,
    train: Vec<TokenizedExample>,
    val: Vec<TokenizedExample>,
    rng: Rng,
}

impl SftTrainer {
    /// `train` and `val` must already fit the model's context (see `tokenize_all`).
    pub fn new(model: Transformer, cfg: SftConfig, train: Vec<TokenizedExample>, val: Vec<TokenizedExample>, total_steps: usize, seed: u64) -> Result<Self, String> {
        if train.is_empty() || val.is_empty() {
            return Err("need at least one training and one validation example that fits the context window".into());
        }
        // A fresh optimizer: fine-tuning starts a new learning phase.
        let opt = AdamW::new(&model.params, cfg.base_lr, cfg.weight_decay);
        Ok(SftTrainer { model, opt, cfg, step: 0, total_steps, train, val, rng: Rng::new(seed) })
    }

    fn seq_len(&self) -> usize {
        self.model.cfg.ctx
    }

    /// Continue a saved run: set the step counter and reseed the batch stream
    /// from it, so the resumed run draws fresh batches (all replicas do the same).
    pub fn resume_at(&mut self, step: usize, seed: u64) {
        self.step = step;
        self.rng = Rng::new(seed.wrapping_add((step as u64).wrapping_mul(7919)));
    }

    pub fn train_step(&mut self) -> f32 {
        let (loss, mut grads) = self.compute_grads(0, 1);
        self.apply_grads(&mut grads);
        loss
    }

    /// Data-parallel half 1 (see `parallel`): draw the whole batch, compute
    /// gradients for worker `shard`'s share only. Every share divides by the
    /// WHOLE batch's number of response tokens (prompts differ in length, so
    /// shares would otherwise be weighted unevenly).
    pub fn compute_grads(&mut self, shard: usize, n_shards: usize) -> (f32, Vec<Matrix>) {
        let seq = self.seq_len();
        let chosen: Vec<usize> = (0..self.cfg.batch).map(|_| self.rng.next_usize(self.train.len())).collect();
        let (x, y) = pack(&self.train, &chosen, seq);
        let counted = y.iter().filter(|&&t| t != IGNORE_TARGET).count();
        let mine = shard_range(self.cfg.batch, shard, n_shards);
        let (xs, ys) = (&x[mine.start * seq..mine.end * seq], &y[mine.start * seq..mine.end * seq]);
        self.model.loss_and_grads_scaled(xs, ys, seq, self.cfg.threads, counted)
    }

    /// Data-parallel half 2: apply the summed gradients.
    pub fn apply_grads(&mut self, grads: &mut [Matrix]) {
        clip_grad_norm(grads, self.cfg.clip);
        self.opt.lr = lr_at(self.step, self.total_steps, self.cfg.warmup, self.cfg.base_lr);
        self.opt.step(&mut self.model.params, grads);
        self.step += 1;
    }

    /// Response-only loss on held-out examples.
    pub fn validate(&self) -> f32 {
        let chosen: Vec<usize> = (0..self.val.len().min(256)).collect();
        let (x, y) = pack(&self.val, &chosen, self.seq_len());
        self.model.loss(&x, &y, self.seq_len())
    }
}

// ---- evaluation ------------------------------------------------------------------------

/// True if `word` appears in `text` as a whole word (ignoring case).
pub fn contains_word(text: &str, word: &str) -> bool {
    let (text, word) = (text.to_ascii_lowercase(), word.to_ascii_lowercase());
    text.match_indices(&word).any(|(i, _)| {
        let before = text[..i].chars().next_back().is_none_or(|c| !c.is_ascii_alphabetic());
        let after = text[i + word.len()..].chars().next().is_none_or(|c| !c.is_ascii_alphabetic());
        before && after
    })
}

/// Generate an answer to `prompt`, stopping at `STOP`. Returns the answer text
/// (without the stop marker) and whether the model actually produced the marker.
pub fn answer(model: &Transformer, tok: &Tokenizer, prompt: &str, max_tokens: usize, temperature: f32, rng: &mut Rng) -> (String, bool) {
    let ids = tok.encode(prompt);
    if ids.is_empty() {
        return (String::new(), false);
    }
    let out = model.sample_until(&ids, max_tokens, temperature, rng, |generated| tok.decode(generated).contains(STOP));
    let text = tok.decode(&out);
    match text.find(STOP) {
        Some(at) => (text[..at].to_string(), true),
        None => (text, false),
    }
}

/// The control for the "uses the word" score: how often does an answer contain
/// a word that was asked in a DIFFERENT question? If the model only guesses
/// common story words without reading the prompt, this is about as high as the
/// real score. Answer i is checked against the word of question i+1.
pub fn chance_rate(answers: &[String], words: &[String]) -> f32 {
    assert_eq!(answers.len(), words.len());
    let n = answers.len();
    let (mut hits, mut counted) = (0, 0);
    for i in 0..n {
        let other = &words[(i + 1) % n];
        if other == &words[i] {
            continue; // same word asked twice: not a valid control pair
        }
        counted += 1;
        hits += contains_word(&answers[i], other) as usize;
    }
    if counted == 0 { 0.0 } else { hits as f32 / counted as f32 }
}

#[derive(Debug, Clone)]
pub struct WordEval {
    pub n: usize,
    /// Fraction of answers that contain the requested word.
    pub uses_word: f32,
    /// Fraction that contain some OTHER question's word (see `chance_rate`).
    pub chance: f32,
    /// Fraction of answers that ended with the stop marker (rather than rambling on).
    pub stops: f32,
    /// A few (prompt, answer) pairs to look at.
    pub samples: Vec<(String, String)>,
}

/// Ask the model `n` held-out "use the word" questions and score the answers.
pub fn evaluate_word_task(model: &Transformer, tok: &Tokenizer, held_out: &[Example], n: usize, temperature: f32, seed: u64) -> WordEval {
    let mut rng = Rng::new(seed);
    let n = n.min(held_out.len());
    let (mut used, mut stopped) = (0, 0);
    let mut samples = Vec::new();
    let (mut answers, mut words) = (Vec::new(), Vec::new());
    for ex in &held_out[..n] {
        let word = ex.prompt.trim_start_matches("Word: ").split('\n').next().unwrap_or("").to_string();
        let (text, stops) = answer(model, tok, &ex.prompt, 60, temperature, &mut rng);
        used += contains_word(&text, &word) as usize;
        stopped += stops as usize;
        if samples.len() < 6 {
            samples.push((ex.prompt.clone(), text.clone()));
        }
        answers.push(text);
        words.push(word);
    }
    WordEval {
        n,
        uses_word: used as f32 / n.max(1) as f32,
        chance: if n > 1 { chance_rate(&answers, &words) } else { 0.0 },
        stops: stopped as f32 / n.max(1) as f32,
        samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TransformerConfig;

    fn tok() -> Tokenizer {
        Tokenizer::train(&"Word: ball\nSentence: A red ball.<|endoftext|> the cat sat on the mat. ".repeat(40), 300)
    }

    #[test]
    fn pack_masks_the_prompt_and_the_padding() {
        // prompt = 3 tokens (10,11,12), response = 3 tokens (20,21,22).
        let ex = TokenizedExample { ids: vec![10, 11, 12, 20, 21, 22], prompt_len: 3 };
        let (x, y) = pack(&[ex], &[0], 8);
        assert_eq!(&x[..5], &[10, 11, 12, 20, 21]); // the model reads everything but the last token
        let m = IGNORE_TARGET;
        // Targets: predicting 11, 12 (inside the prompt) is masked; predicting 20, 21, 22 is learned.
        assert_eq!(y, vec![m, m, 20, 21, 22, m, m, m]); // trailing entries are padding
    }

    #[test]
    fn tokenize_keeps_the_prompt_boundary() {
        let t = tok();
        let ex = Example { prompt: "Word: ball\nSentence:".into(), response: format!(" A red ball.{STOP}") };
        let te = tokenize(&t, &ex);
        assert_eq!(te.prompt_len, t.encode(&ex.prompt).len());
        assert_eq!(t.decode(&te.ids[..te.prompt_len]), ex.prompt);
        assert_eq!(t.decode(&te.ids[te.prompt_len..]), ex.response);
    }

    #[test]
    fn too_long_examples_are_dropped() {
        let t = tok();
        let long = Example { prompt: "Word: x\nSentence:".into(), response: format!("{}{STOP}", " the cat sat".repeat(30)) };
        let short = Example { prompt: "Word: ball\nSentence:".into(), response: format!(" A red ball.{STOP}") };
        let kept = tokenize_all(&t, &[long, short], 40);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn word_examples_are_well_formed_and_reproducible() {
        let stories = "Once upon a time, a little girl named Lily found a shiny red ball in the garden.\nMore text.\n<|endoftext|>\n\
                       Tom and his dog loved to play by the river every sunny morning.\nMore.\n<|endoftext|>\n\
                       Hi.\n<|endoftext|>\n";
        let a = word_examples(stories, 10, &mut Rng::new(1));
        let b = word_examples(stories, 10, &mut Rng::new(1));
        assert_eq!(a, b);
        assert_eq!(a.len(), 2, "the too-short third story is skipped");
        for ex in &a {
            let word = ex.prompt.strip_prefix("Word: ").unwrap().split('\n').next().unwrap();
            assert!(contains_word(&ex.response, word), "{word} should appear in {}", ex.response);
            assert!(ex.response.starts_with(' ') && ex.response.ends_with(STOP));
            assert!(ex.prompt.ends_with("\nSentence:"));
        }
    }

    #[test]
    fn text_without_markers_is_split_into_paragraphs() {
        let text = "The old lighthouse keeper climbed the spiral stairs every single evening.\nHe lit the lamp.\n\n\
                    A small green frog jumped across the muddy garden path at midnight.\n\n";
        let ex = word_examples(text, 10, &mut Rng::new(3));
        assert_eq!(ex.len(), 2);
    }

    #[test]
    fn contains_word_respects_word_boundaries() {
        assert!(contains_word("A red Ball rolled.", "ball"));
        assert!(contains_word("ball", "ball"));
        assert!(!contains_word("He plays football.", "ball"));
        assert!(!contains_word("Balloons!", "ball"));
        assert!(contains_word("the ball, the", "ball"));
    }

    #[test]
    fn chance_control_only_counts_the_other_questions_word() {
        let strs = |xs: &[&str]| xs.iter().map(|s| s.to_string()).collect::<Vec<String>>();
        // Answer i is checked against the word of question i+1 (wrapping around).
        let words = strs(&["ball", "dog", "cat", "dog"]);
        let own = strs(&["a ball", "a dog", "a cat", "a dog"]); // each answer has only its own word
        assert_eq!(chance_rate(&own, &words), 0.0);

        // Answer 0 also mentions "dog", question 1's word: 1 hit out of 4 valid pairs.
        let mut leaky = own.clone();
        leaky[0] = "a ball and a dog".into();
        assert!((chance_rate(&leaky, &words) - 0.25).abs() < 1e-6);

        // Questions 0 and 1 ask for the same word, so pair 0 is not a valid control and is skipped.
        // Left: pair 1 ("dog" answer vs "cat") and pair 2 ("cat" answer vs "dog"). One hit in two.
        let words = strs(&["dog", "dog", "cat"]);
        let answers = strs(&["a dog", "a dog and a cat", "a cat"]);
        assert!((chance_rate(&answers, &words) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn jsonl_parsing() {
        let ok = parse_jsonl("{\"prompt\":\"Hi\",\"response\":\" Hello\"}\n\n{\"prompt\":\"A\",\"response\":\"B\"}\n").unwrap();
        assert_eq!(ok.len(), 2);
        assert_eq!(ok[0].response, format!(" Hello{STOP}"));
        assert!(parse_jsonl("not json").is_err());
        assert!(parse_jsonl("{\"prompt\":\"only\"}").is_err());
        assert!(parse_jsonl("").is_err());
    }

    #[test]
    fn sft_teaches_a_simple_instruction() {
        // Toy task: the prompt contains one of 4 symbols; the response must repeat it 3 times.
        // Ids 1..=4 are symbols, 5 = separator, 6 = stop.
        let cfg = TransformerConfig { vocab_size: 8, ctx: 8, d_model: 16, n_layer: 2, n_head: 2 };
        let model = Transformer::new(cfg, &mut Rng::new(2));
        let make = |s: usize| TokenizedExample { ids: vec![s, 5, s, s, s, 6], prompt_len: 2 };
        let train: Vec<_> = (0..40).map(|i| make(1 + i % 4)).collect();
        let val: Vec<_> = (1..=4).map(make).collect();
        let sft = SftConfig { batch: 8, base_lr: 5e-3, weight_decay: 0.0, warmup: 5, clip: 1.0, threads: 1 };
        let mut t = SftTrainer::new(model, sft, train, val, 150, 1).unwrap();
        let before = t.validate();
        for _ in 0..150 {
            t.train_step();
        }
        let after = t.validate();
        assert!(after < before * 0.2 && after < 0.35, "response loss {before} -> {after}");
        // Greedy-ish generation now follows the instruction.
        for s in 1..=4 {
            let out = t.model.sample_until(&[s, 5], 4, 0.05, &mut Rng::new(9), |_| false);
            assert_eq!(out, vec![s, s, s, 6], "symbol {s}");
        }
    }

    #[test]
    fn parallel_fine_tuning_matches_single_process_even_with_uneven_masks() {
        use crate::parallel::{add_flat, checksum, flat_to_grads, grads_to_flat};
        // Examples with DIFFERENT prompt and response lengths, so shares have different token counts.
        let examples: Vec<TokenizedExample> = (0..30)
            .map(|i| {
                let (p, r) = (1 + i % 3, 1 + i % 4);
                let mut ids: Vec<usize> = (0..p).map(|j| 1 + (i + j) % 5).collect();
                ids.extend((0..r).map(|j| 1 + (i * 2 + j) % 5));
                TokenizedExample { ids, prompt_len: p }
            })
            .collect();
        let make = || {
            let cfg = TransformerConfig { vocab_size: 8, ctx: 10, d_model: 16, n_layer: 1, n_head: 2 };
            let sft = SftConfig { batch: 7, base_lr: 3e-3, weight_decay: 0.01, warmup: 3, clip: 1.0, threads: 1 };
            SftTrainer::new(Transformer::new(cfg, &mut Rng::new(4)), sft, examples.clone(), examples[..5].to_vec(), 50, 11).unwrap()
        };
        let mut solo = make();
        let mut pool: Vec<SftTrainer> = (0..3).map(|_| make()).collect();
        for step in 0..10 {
            let solo_loss = solo.train_step();
            let mut total: Option<Vec<f32>> = None;
            for (i, r) in pool.iter_mut().enumerate() {
                let (loss, grads) = r.compute_grads(i, 3);
                let flat = grads_to_flat(&grads, loss);
                match &mut total {
                    None => total = Some(flat),
                    Some(t) => add_flat(t, &flat),
                }
            }
            let total = total.unwrap();
            for r in pool.iter_mut() {
                let (mut g, _) = flat_to_grads(&r.model.params, &total).unwrap();
                r.apply_grads(&mut g);
            }
            assert!((solo_loss - total.last().unwrap()).abs() < 1e-4, "step {step}");
        }
        assert_eq!(checksum(&pool[0].model.params), checksum(&pool[2].model.params));
        for (a, b) in solo.model.params.iter().zip(&pool[0].model.params) {
            for (x, y) in a.data.iter().zip(&b.data) {
                assert!((x - y).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn training_requires_examples() {
        let cfg = TransformerConfig { vocab_size: 8, ctx: 8, d_model: 16, n_layer: 1, n_head: 2 };
        let model = Transformer::new(cfg, &mut Rng::new(2));
        let sft = SftConfig { batch: 2, base_lr: 1e-3, weight_decay: 0.0, warmup: 1, clip: 1.0, threads: 1 };
        assert!(SftTrainer::new(model, sft, vec![], vec![], 10, 1).is_err());
    }
}
