//! Browser bindings. JavaScript creates a `Session`, then calls `step()` (or the
//! data-parallel pair `compute_grads` / `apply_grads`) from a Web Worker, and
//! `generate()` / `answer()` whenever it wants text.
//!
//! A session moves through phases:
//!   pretrain  learn to continue text                      (`new` or `restore`)
//!   finetune  learn to follow an instruction              (`start_finetune`)
//!   loaded    a model file opened for chatting only       (`from_model`)

mod textbook;
pub use textbook::TokenizerLab;

use serde::{Deserialize, Serialize};
use tlm_core::bpe::Tokenizer;
use tlm_core::dpo::{self, DpoConfig, DpoTrainer};
use tlm_core::optim::AdamW;
use tlm_core::parallel;
use tlm_core::sft::{self, Example, SftConfig, SftTrainer};
use tlm_core::trainer::{TrainConfig, Trainer};
use tlm_core::transformer::{Transformer, TransformerConfig};
use tlm_core::{checkpoint, Rng};
use wasm_bindgen::prelude::*;

/// Runs once when the module loads: make Rust panics readable in the browser console.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// How much text a session keeps (for fine-tuning examples and for resuming).
const MAX_TEXT_BYTES: usize = 8_000_000;

/// Pre-training settings from the page. Missing fields take the defaults below.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(default)]
struct Config {
    vocab_size: usize,
    d_model: usize,
    n_layer: usize,
    n_head: usize,
    ctx: usize,
    batch: usize,
    lr: f32,
    steps: usize,
    seed: u64,
    /// Pre-training uses only the first this-many bytes of the text; the rest is
    /// kept back so fine-tuning questions come from text the model never saw.
    max_bytes: usize,
}

impl Default for Config {
    fn default() -> Self {
        // ctx 64: a fine-tuning example (instruction + answer) must fit in the window.
        Config { vocab_size: 512, d_model: 64, n_layer: 2, n_head: 4, ctx: 64, batch: 16, lr: 3e-3, steps: 2000, seed: 42, max_bytes: 1_000_000 }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(default)]
struct SftSettings {
    steps: usize,
    lr: f32,
    batch: usize,
    /// At most this many examples are built from the text.
    examples: usize,
    seed: u64,
    /// The user's own examples, one JSON object per line: {"prompt": "...", "response": "..."}.
    /// Empty means the built-in "use the word" task. Kept here (not beside it) so that a saved
    /// session and every helper worker rebuild exactly the same examples from the settings alone.
    custom: String,
}

impl Default for SftSettings {
    fn default() -> Self {
        SftSettings { steps: 1500, lr: 1e-3, batch: 16, examples: 20_000, seed: 7, custom: String::new() }
    }
}

/// The most custom example text accepted (a few thousand short examples).
const MAX_CUSTOM_BYTES: usize = 1_000_000;
/// The fewest custom examples worth fine-tuning on (some are held out for scoring).
const MIN_CUSTOM_EXAMPLES: usize = 20;
/// Task names stored in `SftInfo`.
const TASK_WORD: &str = "use-the-word";
const TASK_CUSTOM: &str = "custom";

fn default_task() -> String {
    TASK_WORD.to_string()
}

/// What a fine-tuning run is made from (saved so a resumed session can rebuild it).
#[derive(Serialize, Deserialize, Debug, Clone)]
struct SftInfo {
    settings: SftSettings,
    /// True if there was not enough unseen text, so examples overlap the pre-training text.
    overlap: bool,
    train_examples: usize,
    val_examples: usize,
    template: String,
    /// "use-the-word" (built in, can be checked automatically) or "custom" (the user's own examples).
    #[serde(default = "default_task")]
    task: String,
}

/// Preference-tuning (DPO) settings from the page.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(default)]
struct DpoSettings {
    steps: usize,
    lr: f32,
    /// How strongly to prefer chosen over rejected answers, relative to the starting model.
    beta: f32,
    /// Weight of the supervised term that keeps chosen answers likely (see the dpo module).
    sft_weight: f32,
    batch: usize,
    /// How many prompts to sample answers for.
    prompts: usize,
    /// Answers sampled per prompt.
    k: usize,
    mine_temperature: f32,
    /// Longest answer sampled while mining, in tokens.
    max_new: usize,
    seed: u64,
}

impl Default for DpoSettings {
    fn default() -> Self {
        DpoSettings { steps: 300, lr: 1e-4, beta: 0.3, sft_weight: 4.0, batch: 16, prompts: 1500, k: 4, mine_temperature: 0.7, max_new: 60, seed: 11 }
    }
}

/// What a preference-tuning run is made from.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct DpoInfo {
    settings: DpoSettings,
    mined_prompts: usize,
    mined_samples: usize,
    /// Sampled answers that passed the check (uses the word and finishes properly).
    mined_good: usize,
    pairs: usize,
    train_pairs: usize,
    val_pairs: usize,
}

/// Everything JavaScript needs to save, besides the binary blobs.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Meta {
    version: u32,
    phase: String,
    step: usize,
    total_steps: usize,
    base_steps: usize,
    /// Fine-tuning steps completed (once the model has moved on to preference tuning).
    #[serde(default)]
    ft_steps: usize,
    params: usize,
    vocab_size: usize,
    n_tokens: usize,
    bytes_per_token: f32,
    config: Config,
    sft: Option<SftInfo>,
    #[serde(default)]
    dpo: Option<DpoInfo>,
}

fn validate(c: &Config) -> Result<(), String> {
    let check = |ok: bool, msg: &str| if ok { Ok(()) } else { Err(msg.to_string()) };
    check((257..=4096).contains(&c.vocab_size), "vocab_size must be between 257 and 4096")?;
    check((8..=512).contains(&c.d_model), "d_model must be between 8 and 512")?;
    check((1..=12).contains(&c.n_layer), "n_layer must be between 1 and 12")?;
    check(c.n_head >= 1 && c.d_model % c.n_head == 0, "d_model must be divisible by n_head")?;
    check((4..=256).contains(&c.ctx), "ctx must be between 4 and 256")?;
    check((1..=256).contains(&c.batch), "batch must be between 1 and 256")?;
    check(c.lr > 0.0 && c.lr <= 0.1, "lr must be in (0, 0.1]")?;
    check((1..=1_000_000).contains(&c.steps), "steps must be between 1 and 1,000,000")?;
    check(c.max_bytes >= 10_000, "need at least 10,000 bytes of text")
}

fn validate_sft(s: &SftSettings) -> Result<(), String> {
    let check = |ok: bool, msg: &str| if ok { Ok(()) } else { Err(msg.to_string()) };
    check((1..=1_000_000).contains(&s.steps), "fine-tuning steps must be between 1 and 1,000,000")?;
    check(s.lr > 0.0 && s.lr <= 0.1, "fine-tuning lr must be in (0, 0.1]")?;
    check((1..=256).contains(&s.batch), "fine-tuning batch must be between 1 and 256")?;
    check((50..=100_000).contains(&s.examples), "examples must be between 50 and 100,000")?;
    check(s.custom.len() <= MAX_CUSTOM_BYTES, "the custom examples are too large (at most 1 MB)")
}

fn validate_dpo(s: &DpoSettings) -> Result<(), String> {
    let check = |ok: bool, msg: &str| if ok { Ok(()) } else { Err(msg.to_string()) };
    check((1..=1_000_000).contains(&s.steps), "preference-tuning steps must be between 1 and 1,000,000")?;
    check(s.lr > 0.0 && s.lr <= 0.01, "preference-tuning lr must be in (0, 0.01]")?;
    check(s.beta > 0.0 && s.beta <= 5.0, "beta must be in (0, 5]")?;
    check((0.0..=50.0).contains(&s.sft_weight), "sft_weight must be between 0 and 50")?;
    check((1..=256).contains(&s.batch), "preference-tuning batch must be between 1 and 256")?;
    check((20..=5000).contains(&s.prompts), "prompts must be between 20 and 5,000")?;
    check((2..=16).contains(&s.k), "answers per prompt (k) must be between 2 and 16")?;
    check(s.mine_temperature > 0.0 && s.mine_temperature <= 2.0, "mining temperature must be in (0, 2]")?;
    check((8..=200).contains(&s.max_new), "max_new must be between 8 and 200")
}

fn dpo_config(s: &DpoSettings) -> DpoConfig {
    DpoConfig { batch: s.batch, base_lr: s.lr, beta: s.beta, sft_weight: s.sft_weight, weight_decay: 0.0, warmup: (s.steps / 20).clamp(1, 50), clip: 1.0, threads: 1 }
}

const CUSTOM_NO_DPO: &str = "preference tuning needs an automatic check of which answers are good, and your own examples have none. It is available for the built-in \"use the word\" task.";

/// The fewest preference pairs worth training on.
const MIN_PAIRS: usize = 40;

/// Cut `text` to at most `max` bytes without splitting a multi-byte character.
fn truncate_utf8(text: &str, max: usize) -> &str {
    let mut end = max.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn js_err(msg: impl AsRef<str>) -> JsError {
    JsError::new(msg.as_ref())
}

fn train_config(c: &Config) -> TrainConfig {
    TrainConfig { seq: c.ctx, batch: c.batch, base_lr: c.lr, weight_decay: 0.01, warmup: (c.steps / 20).clamp(1, 100), clip: 1.0, threads: 1 }
}

fn sft_config(s: &SftSettings) -> SftConfig {
    SftConfig { batch: s.batch, base_lr: s.lr, weight_decay: 0.01, warmup: (s.steps / 20).clamp(1, 100), clip: 1.0, threads: 1 }
}

#[derive(PartialEq, Clone, Copy)]
enum Phase {
    Pretrain,
    Finetune,
    Dpo,
    Loaded,
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Phase::Pretrain => "pretrain",
            Phase::Finetune => "finetune",
            Phase::Dpo => "dpo",
            Phase::Loaded => "loaded",
        }
    }
}

#[wasm_bindgen]
pub struct Session {
    tok: Tokenizer,
    cfg: Config,
    /// The training text (up to MAX_TEXT_BYTES).
    text: String,
    bytes_per_token: f32,
    n_tokens: usize,
    /// Steps of pre-training done (kept after the model moves into fine-tuning).
    base_steps: usize,
    pre: Option<Trainer>,
    ft: Option<SftTrainer>,
    ft_info: Option<SftInfo>,
    /// Held-out fine-tuning questions, for scoring.
    ft_val: Vec<Example>,
    loaded: Option<Transformer>,
    /// The prompt layout for `answer`; `{}` is replaced by the user's input.
    template: Option<String>,
    /// Prompts available for mining preference pairs (the fine-tuning training prompts).
    ft_prompts: Vec<String>,
    /// Fine-tuning steps completed (kept after the model moves on to preference tuning).
    ft_done_steps: usize,
    dpo: Option<DpoTrainer>,
    dpo_info: Option<DpoInfo>,
    /// Mining results from each worker's share of the prompts, collected until `start_dpo`.
    mined: Vec<Vec<u8>>,
    /// The scored preference pairs, as bytes, so a run in progress can be saved.
    pairs_bytes: Vec<u8>,
}

impl Session {
    /// A session with a tokenizer and config and nothing else yet.
    fn empty(tok: Tokenizer, cfg: Config, text: String) -> Session {
        Session {
            tok, cfg, text, bytes_per_token: 0.0, n_tokens: 0, base_steps: 0,
            pre: None, ft: None, ft_info: None, ft_val: Vec::new(), loaded: None, template: None,
            ft_prompts: Vec::new(), ft_done_steps: 0, dpo: None, dpo_info: None, mined: Vec::new(), pairs_bytes: Vec::new(),
        }
    }

    fn phase_of(&self) -> Phase {
        if self.dpo.is_some() {
            Phase::Dpo
        } else if self.ft.is_some() {
            Phase::Finetune
        } else if self.pre.is_some() {
            Phase::Pretrain
        } else {
            Phase::Loaded
        }
    }

    /// True when fine-tuned on the user's own examples (which have no automatic checker).
    fn is_custom_task(&self) -> bool {
        self.ft_info.as_ref().is_some_and(|i| i.task == TASK_CUSTOM)
    }

    fn model(&self) -> &Transformer {
        match (&self.dpo, &self.ft, &self.pre, &self.loaded) {
            (Some(t), _, _, _) => &t.model,
            (_, Some(t), _, _) => &t.model,
            (_, _, Some(t), _) => &t.model,
            (_, _, _, Some(m)) => m,
            _ => unreachable!("a session always holds a model"),
        }
    }

    /// Total optimiser steps the current model has been through, across all phases.
    fn total_trained(&self) -> usize {
        match self.phase_of() {
            Phase::Pretrain => self.pre.as_ref().map_or(0, |t| t.step),
            Phase::Finetune => self.base_steps + self.ft.as_ref().map_or(0, |t| t.step),
            Phase::Dpo => self.base_steps + self.ft_done_steps + self.dpo.as_ref().map_or(0, |t| t.step),
            Phase::Loaded => self.base_steps,
        }
    }

    /// Build the pre-training trainer for `model` from the first `max_bytes` of the text.
    fn make_pretrainer(tok: &Tokenizer, cfg: &Config, text: &str, model: Transformer, opt: AdamW, step: usize) -> Result<(Trainer, f32, usize), String> {
        let base = truncate_utf8(text, cfg.max_bytes);
        let ids = tok.encode(base);
        let bytes_per_token = base.len() as f32 / ids.len().max(1) as f32;
        let n = ids.len();
        let trainer = Trainer::new(model, opt, train_config(cfg), &ids, step, cfg.steps, cfg.seed)?;
        Ok((trainer, bytes_per_token, n))
    }

    /// Build fine-tuning examples from the text AFTER the pre-training portion
    /// (cut at a story boundary), so held-out questions are unseen by the model.
    /// Falls back to the whole text, flagged as `overlap`, when too little remains.
    fn build_sft_data(&self, s: &SftSettings) -> Result<(Vec<Example>, Vec<Example>, bool), String> {
        if !s.custom.trim().is_empty() {
            return Self::build_custom_data(s);
        }
        let base_len = truncate_utf8(&self.text, self.cfg.max_bytes).len();
        let rest = &self.text[base_len..];
        let rest = rest.find(sft::STOP).map_or("", |i| &rest[i + sft::STOP.len()..]);
        let mut overlap = false;
        let mut examples = sft::word_examples(rest, s.examples, &mut Rng::new(s.seed));
        if examples.len() < 50 {
            overlap = true;
            examples = sft::word_examples(&self.text, s.examples, &mut Rng::new(s.seed));
        }
        if examples.len() < 50 {
            return Err(format!(
                "only {} usable examples could be built from this text; fine-tuning needs at least 50 story-like paragraphs with a clear first sentence",
                examples.len()
            ));
        }
        let n_val = (examples.len() / 10).max(20).min(examples.len() / 2);
        let val = examples.split_off(examples.len() - n_val);
        Ok((examples, val, overlap))
    }

    /// The user's own examples: parse, shuffle (so the held-out ones are a fair sample), hold some out.
    fn build_custom_data(s: &SftSettings) -> Result<(Vec<Example>, Vec<Example>, bool), String> {
        let mut examples = sft::parse_jsonl(&s.custom)?;
        if examples.len() < MIN_CUSTOM_EXAMPLES {
            return Err(format!(
                "found {} examples; fine-tuning needs at least {MIN_CUSTOM_EXAMPLES} (a few are held out to score the model). Add more lines to your file.",
                examples.len()
            ));
        }
        examples.truncate(s.examples);
        let mut rng = Rng::new(s.seed);
        for i in (1..examples.len()).rev() {
            examples.swap(i, rng.next_usize(i + 1));
        }
        let n_val = (examples.len() / 10).max(2).min(examples.len() / 2);
        let val = examples.split_off(examples.len() - n_val);
        Ok((examples, val, false))
    }

    /// Score a model fine-tuned on the user's own examples: ask it held-out prompts and compare with the
    /// answers the user gave. There is no right-or-wrong check, so this reports how often it ends properly,
    /// how often it matches the reference exactly, and a few side-by-side examples.
    fn eval_custom(&self, n: usize, temperature: f32, seed: u64) -> String {
        let mut rng = Rng::new(seed);
        let n = n.min(self.ft_val.len());
        let (mut stopped, mut exact) = (0usize, 0usize);
        let mut samples = Vec::new();
        for ex in &self.ft_val[..n] {
            let (text, stops) = sft::answer(self.model(), &self.tok, &ex.prompt, 60, temperature, &mut rng);
            let reference = ex.response.strip_suffix(sft::STOP).unwrap_or(&ex.response).to_string();
            stopped += stops as usize;
            exact += (text.trim() == reference.trim()) as usize;
            if samples.len() < 6 {
                samples.push(serde_json::json!({ "prompt": ex.prompt, "answer": text, "reference": reference }));
            }
        }
        let frac = |k: usize| k as f32 / n.max(1) as f32;
        serde_json::json!({ "task": TASK_CUSTOM, "n": n, "stops": frac(stopped), "exact": frac(exact), "samples": samples }).to_string()
    }

    // The logic of the exported methods lives here, returning plain `String` errors.
    // (A `JsError` can only be created inside a browser, so this keeps every error
    // path testable natively; the exported wrappers just convert.)

    fn start_finetune_impl(&mut self, settings_json: &str) -> Result<(), String> {
        let settings: SftSettings = serde_json::from_str(settings_json).map_err(|e| format!("bad settings: {e}"))?;
        validate_sft(&settings)?;
        if self.ft.is_some() || self.dpo.is_some() {
            return Err("this model has already been fine-tuned".into());
        }
        if self.pre.is_none() {
            return Err("only a model you trained here can be fine-tuned".into());
        }
        let (train, val, overlap) = self.build_sft_data(&settings)?;
        self.ft_prompts = train.iter().take(5000).map(|e| e.prompt.clone()).collect();
        let ctx = self.cfg.ctx;
        let (train_tok, val_tok) = (sft::tokenize_all(&self.tok, &train, ctx), sft::tokenize_all(&self.tok, &val, ctx));
        if train_tok.len() * 2 < train.len() {
            return Err(format!("over half of the fine-tuning examples are too long for this model's {ctx}-token window"));
        }
        let pre = self.pre.take().expect("checked above");
        self.base_steps = pre.step;
        let trainer = SftTrainer::new(pre.model, sft_config(&settings), train_tok, val_tok, settings.steps, settings.seed)?;
        // Custom examples carry their own prompt layout, so the model is asked exactly what the user typed.
        let (template, task) = if settings.custom.trim().is_empty() { (sft::WORD_TEMPLATE, TASK_WORD) } else { ("{}", TASK_CUSTOM) };
        self.template = Some(template.to_string());
        self.ft_info = Some(SftInfo { settings, overlap, train_examples: train.len(), val_examples: val.len(), template: template.to_string(), task: task.to_string() });
        self.ft_val = val;
        self.ft = Some(trainer);
        Ok(())
    }

    /// Sample answers for one worker's share of the prompts (`shard` of `n_shards`) and pair
    /// good answers with bad ones. `verify` decides what counts as good.
    fn mine_shard_impl(&self, settings_json: &str, shard: usize, n_shards: usize, verify: dpo::Verifier) -> Result<Vec<u8>, String> {
        let settings: DpoSettings = serde_json::from_str(settings_json).map_err(|e| format!("bad settings: {e}"))?;
        validate_dpo(&settings)?;
        let ft = self.ft.as_ref().ok_or("preference tuning starts from a fine-tuned model: fine-tune first")?;
        if self.is_custom_task() {
            return Err(CUSTOM_NO_DPO.into());
        }
        if n_shards == 0 || shard >= n_shards {
            return Err("shard index out of range".into());
        }
        let prompts = &self.ft_prompts[..self.ft_prompts.len().min(settings.prompts)];
        let mine = parallel::shard_range(prompts.len(), shard, n_shards);
        let (pairs, stats) = dpo::mine_pairs(&ft.model, &self.tok, &prompts[mine.clone()], mine.start, settings.k, settings.mine_temperature, settings.max_new, settings.seed, 1, verify);
        Ok(dpo::encode_mined(&pairs, &stats))
    }

    /// Collect one worker's mining results. Call once per worker, in shard order.
    fn add_mined_impl(&mut self, blob: &[u8]) -> Result<(), String> {
        dpo::decode_mined(blob)?; // reject garbage now, not at training time
        self.mined.push(blob.to_vec());
        Ok(())
    }

    /// Switch from fine-tuning to preference tuning, using the collected mining results.
    fn start_dpo_impl(&mut self, settings_json: &str) -> Result<(), String> {
        // Whatever happens next, these results are used up: a retry must collect fresh ones.
        let blobs = std::mem::take(&mut self.mined);
        if self.is_custom_task() {
            return Err(CUSTOM_NO_DPO.into());
        }
        let settings: DpoSettings = serde_json::from_str(settings_json).map_err(|e| format!("bad settings: {e}"))?;
        validate_dpo(&settings)?;
        if self.ft.is_none() {
            return Err(if self.dpo.is_some() { "this model has already been through preference tuning" } else { "preference tuning starts from a fine-tuned model: fine-tune first" }.into());
        }
        let (mut raw, mut stats) = (Vec::new(), dpo::MineStats::default());
        for blob in blobs {
            let (pairs, s) = dpo::decode_mined(&blob)?;
            raw.extend(pairs);
            stats.prompts += s.prompts;
            stats.samples += s.samples;
            stats.good += s.good;
        }
        if raw.len() < MIN_PAIRS {
            return Err(format!(
                "only {} preference pairs found (need at least {MIN_PAIRS}): {} of {} sampled answers passed the check. Sample more prompts, or fine-tune longer so the model gives mixed results.",
                raw.len(), stats.good, stats.samples
            ));
        }
        stats.pairs = raw.len();
        let ft = self.ft.take().expect("checked above");
        // Score the pairs with the model AS IT IS NOW: the frozen reference.
        let pairs = dpo::build_pairs(&ft.model, &raw, 1);
        let n_val = (pairs.len() / 10).max(10).min(pairs.len() / 2);
        let (train, val) = pairs.split_at(pairs.len() - n_val);
        self.pairs_bytes = dpo::encode_pairs(&pairs);
        self.ft_done_steps = ft.step;
        self.dpo_info = Some(DpoInfo {
            settings: settings.clone(),
            mined_prompts: stats.prompts,
            mined_samples: stats.samples,
            mined_good: stats.good,
            pairs: pairs.len(),
            train_pairs: train.len(),
            val_pairs: val.len(),
        });
        self.dpo = Some(DpoTrainer::new(ft.model, dpo_config(&settings), train.to_vec(), val.to_vec(), settings.steps, settings.seed)?);
        Ok(())
    }

    fn compute_grads_impl(&mut self, shard: usize, n_shards: usize) -> Result<Vec<f32>, String> {
        if n_shards == 0 || shard >= n_shards {
            return Err("shard index out of range".into());
        }
        let (loss, grads) = if let Some(t) = &mut self.dpo {
            t.compute_grads(shard, n_shards)
        } else if let Some(t) = &mut self.ft {
            t.compute_grads(shard, n_shards)
        } else if let Some(t) = &mut self.pre {
            t.compute_grads(shard, n_shards)
        } else {
            return Err("nothing to train".into());
        };
        Ok(parallel::grads_to_flat(&grads, loss))
    }

    fn apply_grads_impl(&mut self, summed: &[f32]) -> Result<(), String> {
        let (mut grads, _) = parallel::flat_to_grads(&self.model().params, summed)?;
        if let Some(t) = &mut self.dpo {
            t.apply_grads(&mut grads);
        } else if let Some(t) = &mut self.ft {
            t.apply_grads(&mut grads);
        } else if let Some(t) = &mut self.pre {
            t.apply_grads(&mut grads);
        } else {
            return Err("nothing to train".into());
        }
        Ok(())
    }

    fn export_optimizer_impl(&self) -> Result<Vec<u8>, String> {
        let (opt, step) = match (&self.dpo, &self.ft, &self.pre) {
            (Some(t), _, _) => (&t.opt, t.step),
            (_, Some(t), _) => (&t.opt, t.step),
            (_, _, Some(t)) => (&t.opt, t.step),
            _ => return Err("a loaded model has no optimizer state".into()),
        };
        checkpoint::optimizer_to_bytes(&self.model().cfg, opt, step).map_err(|e| e.to_string())
    }

    fn meta(&self) -> Meta {
        let (step, total) = match (&self.dpo, &self.ft, &self.pre) {
            (Some(t), _, _) => (t.step, t.total_steps),
            (_, Some(t), _) => (t.step, t.total_steps),
            (_, _, Some(t)) => (t.step, t.total_steps),
            _ => (self.base_steps, self.base_steps),
        };
        Meta {
            version: 1,
            phase: self.phase_of().name().to_string(),
            step,
            total_steps: total,
            base_steps: self.base_steps,
            ft_steps: self.ft_done_steps,
            params: self.model().num_params(),
            vocab_size: self.tok.vocab_size(),
            n_tokens: self.n_tokens,
            bytes_per_token: self.bytes_per_token,
            config: self.cfg.clone(),
            sft: self.ft_info.clone(),
            dpo: self.dpo_info.clone(),
        }
    }
}

#[wasm_bindgen]
impl Session {
    /// Train a tokenizer on `text` and set up a fresh model, ready to pre-train.
    #[wasm_bindgen(constructor)]
    pub fn new(text: &str, config_json: &str) -> Result<Session, JsError> {
        let cfg: Config = serde_json::from_str(config_json).map_err(|e| js_err(format!("bad config: {e}")))?;
        validate(&cfg).map_err(js_err)?;
        let text = truncate_utf8(text, MAX_TEXT_BYTES).to_string();

        let tok = Tokenizer::train(truncate_utf8(&text, cfg.max_bytes), cfg.vocab_size);
        let model_cfg = TransformerConfig { vocab_size: tok.vocab_size(), ctx: cfg.ctx, d_model: cfg.d_model, n_layer: cfg.n_layer, n_head: cfg.n_head };
        let model = Transformer::new(model_cfg, &mut Rng::new(cfg.seed));
        let opt = AdamW::new(&model.params, cfg.lr, 0.01);
        let (trainer, bytes_per_token, n_tokens) = Session::make_pretrainer(&tok, &cfg, &text, model, opt, 0).map_err(js_err)?;
        let mut session = Session::empty(tok, cfg, text);
        session.bytes_per_token = bytes_per_token;
        session.n_tokens = n_tokens;
        session.pre = Some(trainer);
        Ok(session)
    }

    /// Rebuild a saved session. `text` must be the same text it was trained on.
    /// `pairs_bytes` is the output of `export_pairs` (empty unless the session was
    /// in preference tuning).
    pub fn restore(text: &str, meta_json: &str, model_bytes: &[u8], optimizer_bytes: &[u8], tokenizer_json: &str, pairs_bytes: &[u8]) -> Result<Session, JsError> {
        let meta: Meta = serde_json::from_str(meta_json).map_err(|e| js_err(format!("bad saved state: {e}")))?;
        let tok = Tokenizer::from_json(tokenizer_json).map_err(js_err)?;
        let (model, _) = checkpoint::model_from_bytes(model_bytes).map_err(|e| js_err(e.to_string()))?;
        if model.cfg.vocab_size != tok.vocab_size() {
            return Err(js_err("the saved model and tokenizer do not belong together"));
        }
        let text = truncate_utf8(text, MAX_TEXT_BYTES).to_string();
        let cfg = meta.config.clone();
        validate(&cfg).map_err(js_err)?;

        let mut session = Session::empty(tok, cfg.clone(), text);
        session.bytes_per_token = meta.bytes_per_token;
        session.n_tokens = meta.n_tokens;
        session.base_steps = meta.base_steps;
        session.ft_done_steps = meta.ft_steps;
        let cfg_for_opt = model.cfg;
        match meta.phase.as_str() {
            "pretrain" => {
                let mut opt = AdamW::new(&model.params, cfg.lr, 0.01);
                checkpoint::optimizer_from_bytes(optimizer_bytes, &cfg_for_opt, &mut opt).map_err(|e| js_err(e.to_string()))?;
                let (trainer, bpt, n) = Session::make_pretrainer(&session.tok, &cfg, &session.text, model, opt, meta.step).map_err(js_err)?;
                session.bytes_per_token = bpt;
                session.n_tokens = n;
                session.pre = Some(trainer);
            }
            "finetune" => {
                let info = meta.sft.clone().ok_or_else(|| js_err("saved fine-tuning state is missing its settings"))?;
                let (train, val, overlap) = session.build_sft_data(&info.settings).map_err(js_err)?;
                let (train_tok, val_tok) = (sft::tokenize_all(&session.tok, &train, cfg.ctx), sft::tokenize_all(&session.tok, &val, cfg.ctx));
                let mut trainer = SftTrainer::new(model, sft_config(&info.settings), train_tok, val_tok, meta.total_steps, info.settings.seed).map_err(js_err)?;
                let mut opt = AdamW::new(&trainer.model.params, info.settings.lr, 0.01);
                checkpoint::optimizer_from_bytes(optimizer_bytes, &cfg_for_opt, &mut opt).map_err(|e| js_err(e.to_string()))?;
                trainer.opt = opt;
                trainer.resume_at(meta.step, info.settings.seed);
                session.template = Some(info.template.clone());
                session.ft_info = Some(SftInfo { overlap, ..info.clone() });
                session.ft_prompts = train.iter().take(5000).map(|e| e.prompt.clone()).collect();
                session.ft_val = val;
                session.ft = Some(trainer);
            }
            "dpo" => {
                let info = meta.dpo.clone().ok_or_else(|| js_err("saved preference-tuning state is missing its settings"))?;
                let sft_info = meta.sft.clone().ok_or_else(|| js_err("saved preference-tuning state is missing its fine-tuning settings"))?;
                // The held-out questions are rebuilt from the text exactly as fine-tuning built them.
                let (train, val, overlap) = session.build_sft_data(&sft_info.settings).map_err(js_err)?;
                // The pairs carry the reference model's scores, which cannot be recomputed from the
                // partly trained model, so they come from the saved bytes.
                let pairs = dpo::decode_pairs(pairs_bytes).map_err(js_err)?;
                if pairs.len() != info.pairs || info.val_pairs >= pairs.len() {
                    return Err(js_err("the saved preference pairs do not match the saved settings"));
                }
                let (train_pairs, val_pairs) = pairs.split_at(pairs.len() - info.val_pairs);
                let mut trainer = DpoTrainer::new(model, dpo_config(&info.settings), train_pairs.to_vec(), val_pairs.to_vec(), meta.total_steps, info.settings.seed).map_err(js_err)?;
                let mut opt = AdamW::new(&trainer.model.params, info.settings.lr, 0.0);
                checkpoint::optimizer_from_bytes(optimizer_bytes, &cfg_for_opt, &mut opt).map_err(|e| js_err(e.to_string()))?;
                trainer.opt = opt;
                trainer.resume_at(meta.step, info.settings.seed);
                session.template = Some(sft_info.template.clone());
                session.ft_info = Some(SftInfo { overlap, ..sft_info });
                session.ft_prompts = train.iter().take(5000).map(|e| e.prompt.clone()).collect();
                session.ft_val = val;
                session.pairs_bytes = pairs_bytes.to_vec();
                session.dpo_info = Some(info);
                session.dpo = Some(trainer);
            }
            other => return Err(js_err(format!("cannot restore a session in phase \"{other}\""))),
        }
        Ok(session)
    }

    /// Open a downloaded model file for chatting (no training).
    pub fn from_model(model_bytes: &[u8], tokenizer_json: &str, template: Option<String>) -> Result<Session, JsError> {
        let tok = Tokenizer::from_json(tokenizer_json).map_err(js_err)?;
        let (model, step) = checkpoint::model_from_bytes(model_bytes).map_err(|e| js_err(e.to_string()))?;
        if model.cfg.vocab_size != tok.vocab_size() {
            return Err(js_err("the model and tokenizer do not belong together (different vocabulary sizes)"));
        }
        let cfg = Config { vocab_size: model.cfg.vocab_size, d_model: model.cfg.d_model, n_layer: model.cfg.n_layer, n_head: model.cfg.n_head, ctx: model.cfg.ctx, ..Config::default() };
        let mut session = Session::empty(tok, cfg, String::new());
        session.base_steps = step;
        session.loaded = Some(model);
        session.template = template;
        Ok(session)
    }

    /// Switch from pre-training to fine-tuning on a built-in "use the word" task.
    pub fn start_finetune(&mut self, settings_json: &str) -> Result<(), JsError> {
        self.start_finetune_impl(settings_json).map_err(js_err)
    }

    /// Preference tuning, step 1: sample answers for this worker's share of the prompts
    /// (`shard` of `n_shards`) and pair good answers with bad ones. Returns the pairs as bytes.
    pub fn mine_shard(&self, settings_json: &str, shard: usize, n_shards: usize) -> Result<Vec<u8>, JsError> {
        self.mine_shard_impl(settings_json, shard, n_shards, &dpo::uses_word_and_stops).map_err(js_err)
    }

    /// Preference tuning, step 2: hand over each worker's mining result, in shard order.
    pub fn add_mined(&mut self, blob: &[u8]) -> Result<(), JsError> {
        self.add_mined_impl(blob).map_err(js_err)
    }

    /// Preference tuning, step 3: score the collected pairs with the model as it is now
    /// (the frozen reference) and switch to preference training.
    pub fn start_dpo(&mut self, settings_json: &str) -> Result<(), JsError> {
        self.start_dpo_impl(settings_json).map_err(js_err)
    }

    /// The scored preference pairs as bytes (empty outside preference tuning). Needed to resume a run.
    pub fn export_pairs(&self) -> Vec<u8> {
        self.pairs_bytes.clone()
    }

    /// Fraction of held-out preference pairs where the model now favours the chosen answer
    /// more than the reference did (NaN outside preference tuning).
    pub fn preference_accuracy(&self) -> f32 {
        self.dpo.as_ref().map_or(f32::NAN, |t| t.validate().accuracy)
    }

    pub fn phase(&self) -> String {
        self.phase_of().name().to_string()
    }

    /// Run `n` steps of the current phase. Returns the last training loss.
    pub fn step(&mut self, n: usize) -> f32 {
        let mut loss = 0.0;
        for _ in 0..n {
            if let Some(t) = &mut self.dpo {
                if t.step >= t.total_steps {
                    break;
                }
                loss = t.train_step().loss;
            } else if let Some(t) = &mut self.ft {
                if t.step >= t.total_steps {
                    break;
                }
                loss = t.train_step();
            } else if let Some(t) = &mut self.pre {
                if t.step >= t.total_steps {
                    break;
                }
                loss = t.train_step();
            } else {
                break;
            }
        }
        loss
    }

    /// Data-parallel half 1: gradients for worker `shard` of `n_shards`, as one flat
    /// list whose last number is this share's loss. Add the shares of all workers.
    pub fn compute_grads(&mut self, shard: usize, n_shards: usize) -> Result<Vec<f32>, JsError> {
        self.compute_grads_impl(shard, n_shards).map_err(js_err)
    }

    /// Data-parallel half 2: apply the SUM of all workers' flat gradient lists.
    pub fn apply_grads(&mut self, summed: &[f32]) -> Result<(), JsError> {
        self.apply_grads_impl(summed).map_err(js_err)
    }

    /// A fingerprint of the weights: equal on every worker if they are in sync.
    pub fn checksum(&self) -> f64 {
        parallel::checksum(&self.model().params)
    }

    /// Loss on held-out text (pre-training), held-out answers (fine-tuning) or held-out
    /// preference pairs (preference tuning).
    pub fn validate(&self) -> f32 {
        match (&self.dpo, &self.ft, &self.pre) {
            (Some(t), _, _) => t.validate().loss,
            (_, Some(t), _) => t.validate(),
            (_, _, Some(t)) => t.validate(),
            _ => f32::NAN,
        }
    }

    /// Convert a per-token loss to bits per byte (8.0 = knows nothing about bytes).
    pub fn bits_per_byte(&self, loss: f32) -> f32 {
        loss / std::f32::consts::LN_2 / self.bytes_per_token
    }

    pub fn step_count(&self) -> usize {
        match (&self.dpo, &self.ft, &self.pre) {
            (Some(t), _, _) => t.step,
            (_, Some(t), _) => t.step,
            (_, _, Some(t)) => t.step,
            _ => self.base_steps,
        }
    }

    pub fn total_steps(&self) -> usize {
        match (&self.dpo, &self.ft, &self.pre) {
            (Some(t), _, _) => t.total_steps,
            (_, Some(t), _) => t.total_steps,
            (_, _, Some(t)) => t.total_steps,
            _ => 0,
        }
    }

    pub fn num_params(&self) -> usize {
        self.model().num_params()
    }

    pub fn vocab_size(&self) -> usize {
        self.tok.vocab_size()
    }

    pub fn token_count(&self) -> usize {
        self.n_tokens
    }

    pub fn bytes_per_token(&self) -> f32 {
        self.bytes_per_token
    }

    /// Continue `prompt` with up to `n_tokens` new tokens (works in every phase).
    pub fn generate(&self, prompt: &str, n_tokens: usize, temperature: f32, seed: u32) -> Result<String, JsError> {
        if prompt.is_empty() {
            return Err(js_err("prompt must not be empty"));
        }
        if !(temperature > 0.0 && temperature <= 5.0) {
            return Err(js_err("temperature must be in (0, 5]"));
        }
        let ids = self.tok.encode(prompt);
        let out = self.model().sample(&ids, n_tokens.min(500), temperature, &mut Rng::new(seed as u64 + 1));
        Ok(self.tok.decode(&out))
    }

    /// Answer `input` using the fine-tuning template (e.g. a word -> a sentence using it).
    pub fn answer(&self, input: &str, n_tokens: usize, temperature: f32, seed: u32) -> Result<String, JsError> {
        let template = self.template.as_ref().ok_or_else(|| js_err("this model has not been fine-tuned, so it has no instruction format"))?;
        if input.trim().is_empty() {
            return Err(js_err("type something first"));
        }
        if !(temperature > 0.0 && temperature <= 5.0) {
            return Err(js_err("temperature must be in (0, 5]"));
        }
        let prompt = template.replace("{}", input.trim());
        let (text, stopped) = sft::answer(self.model(), &self.tok, &prompt, n_tokens.min(200), temperature, &mut Rng::new(seed as u64 + 1));
        Ok(format!("{}{}", text.trim_start(), if stopped { "" } else { " …" }))
    }

    /// Score the model on held-out questions. For the built-in task returns JSON
    /// { n, uses_word, chance, stops, samples: [[prompt, answer], ...] }; for the user's own examples
    /// { task: "custom", n, stops, exact, samples: [{prompt, answer, reference}, ...] }.
    pub fn eval_word(&self, n: usize, temperature: f32, seed: u32) -> Result<String, JsError> {
        if self.ft_val.is_empty() {
            return Err(js_err("start fine-tuning first: that sets aside the held-out questions"));
        }
        if self.is_custom_task() {
            return Ok(self.eval_custom(n.clamp(1, 200), temperature, seed as u64));
        }
        let e = sft::evaluate_word_task(self.model(), &self.tok, &self.ft_val, n.clamp(1, 500), temperature, seed as u64);
        Ok(serde_json::json!({ "task": TASK_WORD, "n": e.n, "uses_word": e.uses_word, "chance": e.chance, "stops": e.stops, "samples": e.samples }).to_string())
    }

    /// The model as a safetensors file (the same format the `tlm` command-line tool reads).
    pub fn export_model(&self) -> Result<Vec<u8>, JsError> {
        checkpoint::model_to_bytes(self.model(), self.total_trained()).map_err(|e| js_err(e.to_string()))
    }

    /// Adam's memory, needed to resume training exactly.
    pub fn export_optimizer(&self) -> Result<Vec<u8>, JsError> {
        self.export_optimizer_impl().map_err(js_err)
    }

    /// The tokenizer as JSON (save it next to the model as tokenizer.json).
    pub fn export_tokenizer(&self) -> String {
        self.tok.to_json()
    }

    /// Everything else needed to restore this session, as JSON.
    pub fn export_meta(&self) -> String {
        serde_json::to_string(&self.meta()).expect("meta is always serialisable")
    }

    /// The prompt layout used by `answer`, if the model was fine-tuned.
    pub fn template(&self) -> Option<String> {
        self.template.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Enough story-like text for the tokenizer, the trainer and fine-tuning examples.
    fn corpus() -> String {
        let openers = ["a little girl named Lily found a shiny red ball in the garden", "Tom and his brave dog loved to explore the dark forest",
            "the old farmer planted seven golden pumpkins near the river", "a tiny green frog jumped over the muddy puddle every morning",
            "two happy children built a giant castle from sand at the beach"];
        let mut s = String::new();
        for i in 0..600 {
            s.push_str(&format!("Once upon a time, {} number{}. They played all day.\nThe end of part {}.\n<|endoftext|>\n", openers[i % 5], i % 40, i));
        }
        s
    }

    fn small_cfg(steps: usize) -> String {
        format!(r#"{{"vocab_size": 300, "d_model": 16, "n_layer": 1, "n_head": 2, "ctx": 128, "batch": 4, "steps": {steps}, "max_bytes": 20000}}"#)
    }

    #[test]
    fn defaults_are_valid_and_unsafe_settings_rejected() {
        assert!(validate(&Config::default()).is_ok());
        for bad in [
            Config { d_model: 10_000, ..Config::default() },
            Config { n_head: 5, ..Config::default() },
            Config { vocab_size: 100, ..Config::default() },
            Config { lr: 5.0, ..Config::default() },
        ] {
            assert!(validate(&bad).is_err());
        }
        assert!(validate_sft(&SftSettings::default()).is_ok());
        assert!(validate_sft(&SftSettings { examples: 5, ..SftSettings::default() }).is_err());
    }

    #[test]
    fn truncation_never_splits_a_character() {
        let text = "aé😀b";
        for max in 0..=text.len() + 2 {
            let cut = truncate_utf8(text, max);
            assert!(cut.len() <= max && text.starts_with(cut));
        }
    }

    /// `JsError` can only be CREATED inside a browser, but the success paths run fine natively.
    fn ok<T>(r: Result<T, JsError>) -> T {
        match r {
            Ok(v) => v,
            Err(_) => panic!("expected Ok"),
        }
    }

    fn session(steps: usize) -> Session {
        ok(Session::new(&corpus(), &small_cfg(steps)))
    }

    const FT: &str = r#"{"steps": 20, "batch": 4, "examples": 300}"#;

    /// Unwrap a plain-String result, showing the real error if there is one.
    fn fine<T>(r: Result<T, String>) -> T {
        r.unwrap_or_else(|e| panic!("unexpected error: {e}"))
    }

    fn restored(s: &Session) -> Session {
        let (meta, model, opt, tok) = (s.export_meta(), ok(s.export_model()), fine(s.export_optimizer_impl()), s.export_tokenizer());
        ok(Session::restore(&corpus(), &meta, &model, &opt, &tok, &s.export_pairs()))
    }

    #[test]
    fn pretraining_session_restores_exactly() {
        let mut s = session(40);
        assert_eq!(s.phase(), "pretrain");
        s.step(6);
        let r = restored(&s);
        assert_eq!((r.phase(), r.step_count(), r.total_steps()), ("pretrain".to_string(), 6, 40));
        assert_eq!(r.checksum(), s.checksum());
        assert_eq!(r.validate(), s.validate());
        // Adam's memory came back too: the optimizer files are byte-identical.
        assert_eq!(fine(r.export_optimizer_impl()), fine(s.export_optimizer_impl()));
        assert_eq!(ok(r.generate("Once", 12, 0.5, 3)), ok(s.generate("Once", 12, 0.5, 3)));
        // And the restored session keeps training.
        let mut r = r;
        r.step(2);
        assert_eq!(r.step_count(), 8);
    }

    #[test]
    fn fine_tuning_session_restores_exactly_and_answers_the_same() {
        let mut s = session(10);
        s.step(10);
        fine(s.start_finetune_impl(FT));
        assert_eq!((s.phase(), s.step_count(), s.total_steps()), ("finetune".to_string(), 0, 20));
        s.step(7);
        let meta: Meta = serde_json::from_str(&s.export_meta()).unwrap();
        assert!(!meta.sft.as_ref().unwrap().overlap, "examples must come from text the base never saw");
        assert_eq!(meta.base_steps, 10);

        let r = restored(&s);
        assert_eq!((r.phase(), r.step_count()), ("finetune".to_string(), 7));
        assert_eq!(r.checksum(), s.checksum());
        assert_eq!(r.validate(), s.validate());
        assert_eq!(fine(r.export_optimizer_impl()), fine(s.export_optimizer_impl()));
        assert_eq!(ok(r.answer("garden", 30, 0.5, 5)), ok(s.answer("garden", 30, 0.5, 5)));
        // Both held-out sets were rebuilt identically, so scoring agrees.
        assert_eq!(ok(r.eval_word(20, 0.7, 1)), ok(s.eval_word(20, 0.7, 1)));
    }


    // ---- fine-tuning on the user's own examples ----

    /// `n` question/answer lines in the file format the page accepts.
    fn custom_jsonl(n: usize) -> String {
        (0..n).map(|i| serde_json::json!({ "prompt": format!("Q: colour of item {i}?\nA:"), "response": format!(" item {i} is {}", ["red", "blue", "green"][i % 3]) }).to_string()).collect::<Vec<_>>().join("\n")
    }

    fn custom_settings(jsonl: &str) -> String {
        serde_json::json!({ "steps": 20, "batch": 4, "examples": 300, "custom": jsonl }).to_string()
    }

    #[test]
    fn custom_examples_fine_tune_restore_and_score() {
        let mut s = session(10);
        s.step(10);
        fine(s.start_finetune_impl(&custom_settings(&custom_jsonl(60))));
        assert_eq!((s.phase(), s.total_steps()), ("finetune".to_string(), 20));
        s.step(7);
        let meta: Meta = serde_json::from_str(&s.export_meta()).unwrap();
        let info = meta.sft.unwrap();
        assert_eq!((info.task.as_str(), info.template.as_str(), info.overlap), ("custom", "{}", false));
        assert_eq!(info.train_examples + info.val_examples, 60);
        assert!(info.val_examples >= 2 && info.val_examples <= 10, "{}", info.val_examples);
        // The held-out examples are a shuffled sample, not just the last lines of the file.
        assert_eq!(s.ft_val.len(), info.val_examples);
        let last_lines = (60 - info.val_examples..60).map(|i| format!("Q: colour of item {i}?
A:")).collect::<Vec<_>>();
        assert!(!s.ft_val.iter().all(|e| last_lines.contains(&e.prompt)), "held-out examples should be shuffled, not the file's tail");

        // The model is asked exactly what the user typed (no "Word:" wrapper).
        assert!(ok(s.answer("Q: colour of item 3?\nA:", 12, 0.5, 1)).len() < 200);

        // Saving and restoring rebuilds the same examples from the settings alone.
        let r = restored(&s);
        assert_eq!((r.phase(), r.step_count()), ("finetune".to_string(), 7));
        assert_eq!(r.checksum(), s.checksum());
        assert_eq!(r.validate(), s.validate());
        assert_eq!(ok(r.answer("Q: colour of item 4?\nA:", 12, 0.5, 2)), ok(s.answer("Q: colour of item 4?\nA:", 12, 0.5, 2)));
        assert_eq!(ok(r.eval_word(10, 0.7, 1)), ok(s.eval_word(10, 0.7, 1)));

        // Scoring a custom task reports what it can honestly measure, with the reference answers alongside.
        let e: serde_json::Value = serde_json::from_str(&ok(s.eval_word(10, 0.7, 1))).unwrap();
        assert_eq!(e["task"], "custom");
        assert_eq!(e["n"].as_u64().unwrap() as usize, info.val_examples.min(10));
        assert!(e["stops"].as_f64().unwrap() >= 0.0 && e["exact"].as_f64().unwrap() <= 1.0);
        let first = &e["samples"][0];
        assert!(first["prompt"].as_str().unwrap().starts_with("Q: colour of item"));
        assert!(first["reference"].as_str().unwrap().contains(" is "), "{first}");
        assert!(!first["reference"].as_str().unwrap().contains("<|endoftext|>"), "the stop marker is not part of the reference");
        assert!(e.get("uses_word").is_none(), "the word score does not apply to custom examples");
    }

    #[test]
    fn custom_examples_have_no_preference_tuning_and_say_why() {
        let mut s = session(10);
        s.step(10);
        fine(s.start_finetune_impl(&custom_settings(&custom_jsonl(40))));
        s.step(20);
        let err = s.mine_shard_impl(DPO, 0, 1, &even_length).unwrap_err();
        assert!(err.contains("your own examples have none"), "{err}");
        assert!(s.start_dpo_impl(DPO).unwrap_err().contains("your own examples have none"));
        assert_eq!(s.phase(), "finetune", "a refused preference run leaves the model where it was");
    }

    #[test]
    fn custom_example_files_are_checked_with_clear_messages() {
        let mut s = session(10);
        s.step(10);
        let few = s.start_finetune_impl(&custom_settings(&custom_jsonl(5))).unwrap_err();
        assert!(few.contains("found 5 examples") && few.contains("at least 20"), "{few}");
        assert_eq!(s.phase(), "pretrain", "a refused fine-tune leaves the model where it was");

        let bad = format!("{}\nnot json\n", custom_jsonl(30));
        let err = s.start_finetune_impl(&custom_settings(&bad)).unwrap_err();
        assert!(err.contains("line 31"), "{err}");

        let missing = format!("{}\n{}", custom_jsonl(30), r#"{"prompt": "no response here"}"#);
        let err = s.start_finetune_impl(&custom_settings(&missing)).unwrap_err();
        assert!(err.contains("missing string \"response\""), "{err}");

        let huge = "x".repeat(MAX_CUSTOM_BYTES + 1);
        assert!(s.start_finetune_impl(&custom_settings(&huge)).unwrap_err().contains("too large"));

        // Too long for the window: over half dropped is a clear error rather than a model trained on almost nothing.
        let long = (0..40).map(|i| serde_json::json!({ "prompt": "word ".repeat(80), "response": format!(" r{i}") }).to_string()).collect::<Vec<_>>().join("\n");
        assert!(s.start_finetune_impl(&custom_settings(&long)).unwrap_err().contains("too long"));

        // Whitespace-only custom text means "use the built-in task", not an error.
        fine(s.start_finetune_impl(&custom_settings("  \n ")));
        let meta: Meta = serde_json::from_str(&s.export_meta()).unwrap();
        assert_eq!(meta.sft.unwrap().task, "use-the-word");
    }

    #[test]
    fn saves_from_before_custom_examples_still_load_as_the_word_task() {
        let mut s = session(10);
        s.step(10);
        fine(s.start_finetune_impl(FT));
        let mut v: serde_json::Value = serde_json::from_str(&s.export_meta()).unwrap();
        v["sft"].as_object_mut().unwrap().remove("task");
        v["sft"]["settings"].as_object_mut().unwrap().remove("custom");
        let old: Meta = serde_json::from_value(v).unwrap();
        let info = old.sft.unwrap();
        assert_eq!((info.task.as_str(), info.settings.custom.as_str()), ("use-the-word", ""));
    }

    // ---- preference tuning ----

    const DPO: &str = r#"{"steps": 12, "batch": 4, "prompts": 70, "k": 4, "lr": 0.001, "max_new": 14}"#;

    /// An arbitrary but deterministic verifier, so that both good and bad answers occur even
    /// from a barely trained model (which never produces a proper sentence).
    fn even_length(text: &str, _stopped: bool, _word: &str) -> bool {
        text.len() % 2 == 0
    }

    /// A session that has been pre-trained, fine-tuned, mined (in `shards` pieces) and switched to DPO.
    fn dpo_session(shards: usize) -> Session {
        let mut s = session(10);
        s.step(10);
        fine(s.start_finetune_impl(FT));
        s.step(5);
        for shard in 0..shards {
            let blob = fine(s.mine_shard_impl(DPO, shard, shards, &even_length));
            fine(s.add_mined_impl(&blob));
        }
        fine(s.start_dpo_impl(DPO));
        s
    }

    #[test]
    fn mining_in_pieces_gives_the_same_pairs_as_mining_at_once() {
        let mut s = session(10);
        s.step(10);
        fine(s.start_finetune_impl(FT));
        let whole = fine(s.mine_shard_impl(DPO, 0, 1, &even_length));
        let (whole_pairs, whole_stats) = dpo::decode_mined(&whole).unwrap();
        assert!(whole_pairs.len() > 40, "the arbitrary verifier should split most prompts: {}", whole_pairs.len());
        let (mut joined, mut samples) = (Vec::new(), 0);
        for shard in 0..3 {
            let (pairs, stats) = dpo::decode_mined(&fine(s.mine_shard_impl(DPO, shard, 3, &even_length))).unwrap();
            joined.extend(pairs);
            samples += stats.samples;
        }
        assert_eq!(samples, whole_stats.samples);
        assert_eq!(joined.len(), whole_pairs.len());
        for (a, b) in joined.iter().zip(&whole_pairs) {
            assert_eq!((&a.prompt_ids, &a.chosen, &a.rejected), (&b.prompt_ids, &b.chosen, &b.rejected));
        }
    }

    #[test]
    fn preference_tuning_starts_trains_and_restores_exactly() {
        let mut s = dpo_session(2);
        assert_eq!((s.phase(), s.step_count(), s.total_steps()), ("dpo".to_string(), 0, 12));
        let meta: Meta = serde_json::from_str(&s.export_meta()).unwrap();
        let info = meta.dpo.clone().unwrap();
        assert!(info.pairs >= MIN_PAIRS && info.train_pairs + info.val_pairs == info.pairs);
        assert_eq!((meta.base_steps, meta.ft_steps), (10, 5));
        assert!(s.pairs_bytes.len() > 4 && s.export_pairs() == s.pairs_bytes);
        assert!(s.preference_accuracy() == 0.0, "no movement yet, so no pair is favoured over the reference");

        let before = s.validate();
        assert!((before - std::f32::consts::LN_2).abs() < 1e-3, "starts at ln 2, got {before}");
        s.step(9);
        assert_eq!(s.step_count(), 9);
        assert!(s.validate() < before, "held-out preference loss should fall");

        // Save in the middle of preference tuning, then restore: nothing may change.
        let r = restored(&s);
        assert_eq!((r.phase(), r.step_count(), r.total_steps()), ("dpo".to_string(), 9, 12));
        assert_eq!(r.checksum(), s.checksum());
        assert_eq!(r.validate(), s.validate());
        assert_eq!(r.preference_accuracy(), s.preference_accuracy());
        assert_eq!(fine(r.export_optimizer_impl()), fine(s.export_optimizer_impl()));
        assert_eq!(r.export_pairs(), s.export_pairs());
        assert_eq!(ok(r.answer("garden", 30, 0.5, 5)), ok(s.answer("garden", 30, 0.5, 5)));
        assert_eq!(ok(r.eval_word(20, 0.7, 1)), ok(s.eval_word(20, 0.7, 1)), "the held-out questions were rebuilt identically");
        let mut r = r;
        r.step(10);
        assert_eq!(r.step_count(), 12, "stops at the planned total");
        // The exported model records every phase's steps: 10 pre-training + 5 fine-tuning + 12 preference.
        let (_, trained) = checkpoint::model_from_bytes(&ok(r.export_model())).unwrap();
        assert_eq!(trained, 27);
    }

    #[test]
    fn preference_training_in_parallel_matches_one_session_alone() {
        let (mut a, mut b, mut solo) = (dpo_session(1), dpo_session(1), dpo_session(1));
        assert_eq!(a.checksum(), b.checksum());
        for _ in 0..3 {
            let (ga, gb) = (fine(a.compute_grads_impl(0, 2)), fine(b.compute_grads_impl(1, 2)));
            let sum: Vec<f32> = ga.iter().zip(&gb).map(|(x, y)| x + y).collect();
            fine(a.apply_grads_impl(&sum));
            fine(b.apply_grads_impl(&sum));
            solo.step(1);
        }
        assert_eq!(a.checksum(), b.checksum(), "workers stay identical");
        assert!((a.checksum() - solo.checksum()).abs() < 1e-2 * solo.checksum().abs().max(1.0));
    }

    #[test]
    fn preference_tuning_refuses_bad_starts_with_clear_messages() {
        // Before fine-tuning there is nothing to mine from.
        let mut s = session(5);
        s.step(5);
        assert!(s.mine_shard_impl(DPO, 0, 1, &even_length).unwrap_err().contains("fine-tune first"));
        assert!(s.start_dpo_impl(DPO).unwrap_err().contains("fine-tune first"));

        fine(s.start_finetune_impl(FT));
        // The real verifier needs sentences that finish properly, which an untrained model never writes.
        let blob = fine(s.mine_shard_impl(DPO, 0, 1, &dpo::uses_word_and_stops));
        fine(s.add_mined_impl(&blob));
        let err = s.start_dpo_impl(DPO).unwrap_err();
        assert!(err.contains("only 0 preference pairs"), "{err}");
        assert!(s.ft.is_some(), "a failed start must leave the session in fine-tuning");

        assert!(s.add_mined_impl(&[1, 2, 3]).is_err(), "garbage mining data is rejected");
        assert!(s.start_dpo_impl(r#"{"beta": -1}"#).is_err());
        assert!(s.mined.is_empty(), "a failed start must not keep stale mining results (a retry would duplicate pairs)");
        assert!(s.mine_shard_impl(DPO, 3, 2, &even_length).is_err(), "shard out of range");
        assert!(s.mine_shard_impl(r#"{"k": 1}"#, 0, 1, &even_length).is_err(), "k must be at least 2");

        // Once in preference tuning, fine-tuning and a second DPO start are refused.
        let mut d = dpo_session(1);
        assert!(d.start_finetune_impl(FT).unwrap_err().contains("already"));
        assert!(d.start_dpo_impl(DPO).unwrap_err().contains("already"));
    }

    #[test]
    fn an_old_saved_state_without_the_new_fields_still_loads() {
        // Meta written before preference tuning existed has no ft_steps / dpo.
        let m = Meta { version: 1, phase: "pretrain".into(), step: 1, total_steps: 2, base_steps: 0, ft_steps: 0, params: 5, vocab_size: 300, n_tokens: 9, bytes_per_token: 2.0, config: Config::default(), sft: None, dpo: None };
        let mut value: serde_json::Value = serde_json::to_value(&m).unwrap();
        value.as_object_mut().unwrap().remove("ft_steps");
        value.as_object_mut().unwrap().remove("dpo");
        let old: Meta = serde_json::from_value(value).unwrap();
        assert_eq!((old.ft_steps, old.dpo.is_none()), (0, true));
    }

    #[test]
    fn a_loaded_model_chats_but_cannot_train() {
        let mut s = session(5);
        s.step(5);
        let loaded = ok(Session::from_model(&ok(s.export_model()), &s.export_tokenizer(), None));
        assert_eq!(loaded.phase(), "loaded");
        assert_eq!(ok(loaded.generate("Once", 10, 0.5, 2)), ok(s.generate("Once", 10, 0.5, 2)));
        assert!(loaded.export_optimizer_impl().is_err());
        let mut loaded = loaded;
        assert_eq!(loaded.step(5), 0.0); // nothing to train
        assert!(loaded.compute_grads_impl(0, 1).is_err());
    }

    #[test]
    fn two_sessions_training_in_parallel_match_one_session_training_alone() {
        let (mut a, mut b, mut solo) = (session(30), session(30), session(30));
        for _ in 0..4 {
            let (ga, gb) = (fine(a.compute_grads_impl(0, 2)), fine(b.compute_grads_impl(1, 2)));
            let sum: Vec<f32> = ga.iter().zip(&gb).map(|(x, y)| x + y).collect();
            fine(a.apply_grads_impl(&sum));
            fine(b.apply_grads_impl(&sum));
            solo.step(1);
        }
        assert_eq!(a.checksum(), b.checksum(), "the two workers must stay identical");
        assert_eq!(a.step_count(), 4);
        assert!((a.checksum() - solo.checksum()).abs() < 1e-2 * solo.checksum().abs().max(1.0), "{} vs {}", a.checksum(), solo.checksum());
        assert!(a.apply_grads_impl(&[1.0, 2.0]).is_err(), "wrong-sized gradients are rejected");
    }

    #[test]
    fn fine_tuning_questions_come_from_text_after_the_pretraining_portion() {
        let s = session(5);
        let (train, val, overlap) = s.build_sft_data(&SftSettings { examples: 400, ..SftSettings::default() }).unwrap();
        assert!(!overlap && train.len() > 50 && !val.is_empty());
        // Story i's closing line is "The end of part i." Pre-training saw only the first `max_bytes` bytes.
        let seen = truncate_utf8(&s.text, s.cfg.max_bytes);
        let seen_parts = seen.matches("The end of part").count();
        assert!(seen_parts > 10 && seen_parts < 600);
        // Count how many examples there are versus stories not seen: they must fit in the unseen stories.
        assert!(train.len() + val.len() <= 600 - seen_parts, "{} examples vs {} unseen stories", train.len() + val.len(), 600 - seen_parts);
    }

    #[test]
    fn too_little_text_for_examples_is_a_clear_error() {
        let text = "just a few words, not stories at all".repeat(5);
        let tok = Tokenizer::train(&text, 260);
        let cfg = Config { max_bytes: 10_000, ..Config::default() };
        let s = Session::empty(tok, cfg, text);
        let err = s.build_sft_data(&SftSettings::default()).unwrap_err();
        assert!(err.contains("at least 50"), "{err}");
    }

    #[test]
    fn meta_round_trips_as_json() {
        let m = Meta {
            version: 1, phase: "finetune".into(), step: 7, total_steps: 100, base_steps: 50, ft_steps: 0, dpo: None, params: 123, vocab_size: 512, n_tokens: 999, bytes_per_token: 2.4,
            config: Config::default(),
            sft: Some(SftInfo { settings: SftSettings::default(), overlap: false, train_examples: 10, val_examples: 2, template: "Word: {}\nSentence:".into(), task: "use-the-word".into() }),
        };
        let back: Meta = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!((back.step, back.base_steps, back.sft.unwrap().train_examples), (7, 50, 10));
    }
}
