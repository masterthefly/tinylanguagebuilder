//! `tlm`: train, resume and run tiny language models on the CPU.
//!
//!   tlm train    --data <text file> --out <run dir> [options]
//!   tlm resume   --out <run dir> [--steps <new total>]
//!   tlm generate --model <run dir> [--prompt "text"]
//!   tlm info     --model <run dir>

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;
use tlm_core::bpe::Tokenizer;
use tlm_core::checkpoint;
use tlm_core::dpo::{self, DpoConfig, DpoTrainer};
use tlm_core::optim::AdamW;
use tlm_core::sft::{self, SftConfig, SftTrainer};
use tlm_core::trainer::{TrainConfig, Trainer};
use tlm_core::transformer::{Transformer, TransformerConfig};
use tlm_core::Rng;

type Result<T> = std::result::Result<T, String>;

/// Everything needed to repeat or continue a run. Saved as run.json.
#[derive(Serialize, Deserialize, Clone)]
struct RunConfig {
    data_path: String,
    data_bytes: usize,
    vocab_size: usize,
    seq: usize,
    batch: usize,
    lr: f32,
    weight_decay: f32,
    warmup: usize,
    clip: f32,
    seed: u64,
    steps: usize,
    eval_every: usize,
    save_every: usize,
}

// ---- tiny argument parser (no dependencies) --------------------------------------

struct Args(HashMap<String, String>);

impl Args {
    fn parse(raw: &[String], allowed: &[&str]) -> Result<Args> {
        let mut map = HashMap::new();
        let mut i = 0;
        while i < raw.len() {
            let key = raw[i].strip_prefix("--").ok_or_else(|| format!("unexpected argument \"{}\" (options look like --name value)", raw[i]))?;
            if !allowed.contains(&key) {
                return Err(format!("unknown option --{key}. Allowed: {}", allowed.iter().map(|a| format!("--{a}")).collect::<Vec<_>>().join(" ")));
            }
            let value = raw.get(i + 1).ok_or_else(|| format!("--{key} needs a value"))?;
            map.insert(key.to_string(), value.clone());
            i += 2;
        }
        Ok(Args(map))
    }

    fn get<T: FromStr>(&self, key: &str, default: T) -> Result<T> {
        match self.0.get(key) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("--{key}: \"{v}\" is not a valid value")),
        }
    }

    fn maybe(&self, key: &str) -> Option<&String> {
        self.0.get(key)
    }

    fn require(&self, key: &str) -> Result<String> {
        self.0.get(key).cloned().ok_or_else(|| format!("missing required option --{key}"))
    }
}

fn io_err(what: &str, e: std::io::Error) -> String {
    format!("{what}: {e}")
}

// ---- helpers ----------------------------------------------------------------------

fn load_text(path: &str, bytes: usize) -> Result<String> {
    let raw = std::fs::read(path).map_err(|e| io_err(&format!("cannot read {path}"), e))?;
    Ok(String::from_utf8_lossy(&raw[..bytes.min(raw.len())]).into_owned())
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn train_config(run: &RunConfig, threads: usize) -> TrainConfig {
    TrainConfig { seq: run.seq, batch: run.batch, base_lr: run.lr, weight_decay: run.weight_decay, warmup: run.warmup, clip: run.clip, threads }
}

fn save_checkpoint(dir: &Path, t: &Trainer) -> Result<()> {
    checkpoint::save_model(dir, &t.model, t.step).map_err(|e| io_err("saving model", e))?;
    checkpoint::save_optimizer(dir, &t.model.cfg, &t.opt, t.step).map_err(|e| io_err("saving optimizer", e))
}

/// The loop shared by `train` and `resume`.
fn run_training(t: &mut Trainer, run: &RunConfig, dir: &Path, bytes_per_token: f32) -> Result<()> {
    let to_bpb = |loss: f32| loss / std::f32::consts::LN_2 / bytes_per_token;
    let start = Instant::now();
    let first = t.step;
    println!("parameters: {}   tokens/step: {}   threads: {}", t.model.num_params(), run.batch * run.seq, t.cfg.threads);
    let v0 = t.validate();
    println!("step {:>6}  val {v0:.3}  ({:.2} bits/byte)", t.step, to_bpb(v0));

    while t.step < t.total_steps {
        let loss = t.train_step();
        let done = t.step == t.total_steps;
        if t.step % run.eval_every == 0 || done {
            let val = t.validate();
            println!(
                "step {:>6}  train {loss:.3}  val {val:.3}  ({:.2} bits/byte)  [{:.0}s, {:.0} steps/s]",
                t.step,
                to_bpb(val),
                start.elapsed().as_secs_f32(),
                (t.step - first) as f32 / start.elapsed().as_secs_f32().max(1e-3)
            );
        }
        if t.step % run.save_every == 0 || done {
            save_checkpoint(dir, t)?;
        }
    }
    println!("saved checkpoint to {}", dir.display());
    Ok(())
}

// ---- commands ----------------------------------------------------------------------

fn cmd_train(raw: &[String]) -> Result<()> {
    let args = Args::parse(raw, &[
        "data", "out", "steps", "vocab", "d-model", "layers", "heads", "ctx", "batch", "lr",
        "bytes", "threads", "save-every", "eval-every", "seed", "weight-decay",
    ])?;
    let out = args.require("out")?;
    let dir = Path::new(&out);
    if dir.join("run.json").exists() {
        return Err(format!("{out} already holds a run. Use `tlm resume --out {out}` to continue it, or pick a new --out."));
    }
    let ctx: usize = args.get("ctx", 32)?;
    let run = RunConfig {
        data_path: args.require("data")?,
        data_bytes: args.get("bytes", 4_000_000)?,
        vocab_size: args.get("vocab", 512)?,
        seq: ctx,
        batch: args.get("batch", 16)?,
        lr: args.get("lr", 3e-3)?,
        weight_decay: args.get("weight-decay", 0.01)?,
        warmup: 100,
        clip: 1.0,
        seed: args.get("seed", 42)?,
        steps: args.get("steps", 3000)?,
        eval_every: args.get("eval-every", 250)?,
        save_every: args.get("save-every", 500)?,
    };
    if run.vocab_size < 257 {
        return Err("--vocab must be at least 257 (256 bytes + at least one merge)".into());
    }
    let (d_model, n_head): (usize, usize) = (args.get("d-model", 64)?, args.get("heads", 4)?);
    if n_head == 0 || d_model % n_head != 0 {
        return Err("--d-model must be divisible by --heads".into());
    }
    let threads = args.get("threads", default_threads())?;

    let text = load_text(&run.data_path, run.data_bytes)?;
    println!("training tokenizer on {} bytes...", text.len());
    let tok = Tokenizer::train(&text, run.vocab_size);
    let ids = tok.encode(&text);
    let bytes_per_token = text.len() as f32 / ids.len() as f32;
    println!("{} tokens ({bytes_per_token:.2} bytes/token)", ids.len());

    let cfg = TransformerConfig { vocab_size: tok.vocab_size(), ctx, d_model, n_layer: args.get("layers", 2)?, n_head };
    let model = Transformer::new(cfg, &mut Rng::new(run.seed));
    let opt = AdamW::new(&model.params, run.lr, run.weight_decay);
    let mut trainer = Trainer::new(model, opt, train_config(&run, threads), &ids, 0, run.steps, run.seed)?;

    std::fs::create_dir_all(dir).map_err(|e| io_err("creating run directory", e))?;
    checkpoint::save_tokenizer(dir, &tok).map_err(|e| io_err("saving tokenizer", e))?;
    let json = serde_json::to_string_pretty(&run).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("run.json"), json).map_err(|e| io_err("saving run.json", e))?;

    run_training(&mut trainer, &run, dir, bytes_per_token)
}

fn cmd_resume(raw: &[String]) -> Result<()> {
    let args = Args::parse(raw, &["out", "steps", "threads"])?;
    let out = args.require("out")?;
    let dir = Path::new(&out);
    let json = std::fs::read_to_string(dir.join("run.json")).map_err(|e| io_err(&format!("{out} has no run.json (was it made by `tlm train`?)"), e))?;
    let mut run: RunConfig = serde_json::from_str(&json).map_err(|e| format!("bad run.json: {e}"))?;
    run.steps = args.get("steps", run.steps)?;
    let threads = args.get("threads", default_threads())?;

    let (model, model_step) = checkpoint::load_model(dir).map_err(|e| io_err("loading model", e))?;
    let tok = checkpoint::load_tokenizer(dir).map_err(|e| io_err("loading tokenizer", e))?;
    let mut opt = AdamW::new(&model.params, run.lr, run.weight_decay);
    let opt_step = checkpoint::load_optimizer(dir, &model.cfg, &mut opt).map_err(|e| io_err("loading optimizer", e))?;
    if opt_step != model_step {
        return Err(format!("model is at step {model_step} but optimizer is at step {opt_step}: the checkpoint was interrupted mid-save"));
    }
    if model_step >= run.steps {
        return Err(format!("already at step {model_step}; pass --steps with a larger total to train further"));
    }

    // Same data, same bytes, same tokenizer => same train/validation split as before.
    let text = load_text(&run.data_path, run.data_bytes)?;
    let ids = tok.encode(&text);
    let bytes_per_token = text.len() as f32 / ids.len() as f32;
    println!("resuming {out} from step {model_step} to {}", run.steps);

    let mut trainer = Trainer::new(model, opt, train_config(&run, threads), &ids, model_step, run.steps, run.seed)?;
    let json = serde_json::to_string_pretty(&run).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("run.json"), json).map_err(|e| io_err("saving run.json", e))?;
    run_training(&mut trainer, &run, dir, bytes_per_token)
}

fn cmd_generate(raw: &[String]) -> Result<()> {
    let args = Args::parse(raw, &["model", "prompt", "tokens", "temperature", "seed"])?;
    let dir = args.require("model")?;
    let dir = Path::new(&dir);
    let (model, step) = checkpoint::load_model(dir).map_err(|e| io_err("loading model", e))?;
    let tok = checkpoint::load_tokenizer(dir).map_err(|e| io_err("loading tokenizer", e))?;
    if tok.vocab_size() != model.cfg.vocab_size {
        return Err("tokenizer and model disagree on vocabulary size".into());
    }
    let n_tokens: usize = args.get("tokens", 100)?;
    let temperature: f32 = args.get("temperature", 0.8)?;
    if temperature <= 0.0 {
        return Err("--temperature must be greater than 0".into());
    }
    let mut rng = Rng::new(args.get("seed", 1234)?);

    let mut complete = |prompt: &str| {
        let ids = tok.encode(prompt);
        let out = model.sample(&ids, n_tokens, temperature, &mut rng);
        println!("{prompt}{}", tok.decode(&out));
    };

    if let Some(prompt) = args.maybe("prompt") {
        if prompt.is_empty() {
            return Err("--prompt must not be empty".into());
        }
        complete(prompt);
        return Ok(());
    }

    println!("model at step {step}. This is a base model: it CONTINUES text rather than answering.");
    println!("Type the start of a story and press Enter (empty line to quit).\n");
    let stdin = std::io::stdin();
    loop {
        print!("> ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            break;
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            break;
        }
        complete(line);
        println!();
    }
    Ok(())
}

fn cmd_info(raw: &[String]) -> Result<()> {
    let args = Args::parse(raw, &["model"])?;
    let dir = args.require("model")?;
    let (model, step) = checkpoint::load_model(Path::new(&dir)).map_err(|e| io_err("loading model", e))?;
    let c = model.cfg;
    println!("trained steps: {step}");
    println!("parameters:    {}", model.num_params());
    println!("vocab size:    {}", c.vocab_size);
    println!("context:       {} tokens", c.ctx);
    println!("d_model:       {}   layers: {}   heads: {}", c.d_model, c.n_layer, c.n_head);
    println!("file size:     {} KB", std::fs::metadata(Path::new(&dir).join("model.safetensors")).map(|m| m.len() / 1024).unwrap_or(0));
    Ok(())
}

/// Saved next to a fine-tuned model so `tlm chat` knows how to talk to it.
#[derive(Serialize, Deserialize)]
struct SftMeta {
    base: String,
    task: String,
    /// The prompt layout; `{}` is replaced by what the user types.
    template: String,
    stop: String,
    steps: usize,
    train_examples: usize,
    word_rate_before: Option<f32>,
    word_rate_after: Option<f32>,
    // How the examples were built, so a later stage (DPO) can rebuild the SAME train/held-out
    // split. Missing in files written by older versions.
    #[serde(default)]
    data_path: Option<String>,
    #[serde(default)]
    text_start: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    max_examples: Option<usize>,
    #[serde(default)]
    dpo: Option<DpoMeta>,
}

/// Recorded when a model has also been through preference training.
#[derive(Serialize, Deserialize, Clone)]
struct DpoMeta {
    beta: f32,
    steps: usize,
    pairs: usize,
    lr: f32,
    word_rate_before: f32,
    word_rate_after: f32,
}

/// Where the instruction examples start in the data file.
enum StartAt<'a> {
    /// At this byte offset (recorded by an earlier run).
    Offset(usize),
    /// Right after the text this base model was pre-trained on, so held-out questions are unseen.
    AfterBaseRun(&'a Path),
}

/// How many of `n` examples are held out for scoring.
fn held_out_count(n: usize) -> usize {
    (n / 10).max(20).min(n / 2)
}

/// Build the "use the word" examples from a stories file. Returns the examples (the last
/// `held_out_count` of them are the held-out questions) and the byte offset they start at.
fn build_word_dataset(data: &str, start: StartAt, max_examples: usize, seed: u64) -> Result<(Vec<sft::Example>, usize)> {
    let raw = std::fs::read(data).map_err(|e| io_err(&format!("cannot read {data}"), e))?;
    let text = String::from_utf8_lossy(&raw).into_owned();
    let mut start_at = match start {
        StartAt::Offset(o) => o.min(text.len()),
        StartAt::AfterBaseRun(base_dir) => {
            // If the base model was pre-trained on the start of this same file, continue after that.
            let run: Option<RunConfig> = std::fs::read_to_string(base_dir.join("run.json")).ok().and_then(|j| serde_json::from_str(&j).ok());
            match run {
                Some(r) if Path::new(&r.data_path) == Path::new(data) => {
                    let mut cut = r.data_bytes.min(text.len());
                    while !text.is_char_boundary(cut) {
                        cut += 1;
                    }
                    // Continue at the next story boundary so no story is cut in half.
                    let s = text[cut..].find(sft::STOP).map_or(text.len(), |i| cut + i + sft::STOP.len());
                    println!("the base model was pre-trained on the first {} bytes of this file; building examples from the rest", r.data_bytes);
                    s
                }
                _ => 0,
            }
        }
    };
    while !text.is_char_boundary(start_at) {
        start_at += 1;
    }
    let examples = sft::word_examples(&text[start_at..], max_examples, &mut Rng::new(seed));
    Ok((examples, start_at))
}

fn print_eval(label: &str, e: &sft::WordEval) {
    println!(
        "  {label:<7} uses the requested word: {:>5.1}%   (control, another question's word: {:>5.1}%)   ends properly: {:>5.1}%",
        e.uses_word * 100.0, e.chance * 100.0, e.stops * 100.0
    );
}

fn cmd_sft(raw: &[String]) -> Result<()> {
    let args = Args::parse(raw, &[
        "base", "out", "data", "steps", "lr", "batch", "examples", "eval-n", "temperature", "threads", "seed",
    ])?;
    let (base, out) = (args.require("base")?, args.require("out")?);
    let (base_dir, out_dir) = (Path::new(&base), Path::new(&out));
    if out_dir.join("sft.json").exists() {
        return Err(format!("{out} already holds a fine-tuned model; choose a new --out"));
    }
    let data = args.require("data")?;
    let steps: usize = args.get("steps", 1500)?;
    let seed: u64 = args.get("seed", 7)?;
    let temperature: f32 = args.get("temperature", 0.7)?;
    let eval_n: usize = args.get("eval-n", 100)?;
    let threads = args.get("threads", default_threads())?;

    let (model, base_step) = checkpoint::load_model(base_dir).map_err(|e| io_err("loading base model", e))?;
    let tok = checkpoint::load_tokenizer(base_dir).map_err(|e| io_err("loading tokenizer", e))?;
    let ctx = model.cfg.ctx;

    // Build the instruction dataset. (If the base model was pre-trained on the start of this same
    // file, the examples come from the text AFTER that point, so held-out questions are unseen.)
    let is_jsonl = data.ends_with(".jsonl");
    let max_examples: usize = args.get("examples", 20_000)?;
    let (examples, text_start) = if is_jsonl {
        let raw_text = std::fs::read(&data).map_err(|e| io_err(&format!("cannot read {data}"), e))?;
        (sft::parse_jsonl(&String::from_utf8_lossy(&raw_text))?, 0)
    } else {
        build_word_dataset(&data, StartAt::AfterBaseRun(base_dir), max_examples, seed)?
    };
    if examples.len() < 50 {
        return Err(format!("only {} usable examples; need at least 50", examples.len()));
    }
    let (train_ex, val_ex) = examples.split_at(examples.len() - held_out_count(examples.len()));
    let train = sft::tokenize_all(&tok, train_ex, ctx);
    let val = sft::tokenize_all(&tok, val_ex, ctx);
    println!(
        "{} examples ({} train, {} held out). {} of {} training examples fit the model's {ctx}-token context.",
        examples.len(), train_ex.len(), val_ex.len(), train.len(), train_ex.len()
    );
    if train.len() * 2 < train_ex.len() {
        println!("warning: over half the examples are too long for this model's context window and were dropped.");
    }
    println!("example:  {:?}", format!("{}{}", train_ex[0].prompt, train_ex[0].response));

    // Score the base model on the held-out prompts BEFORE any fine-tuning.
    let score = !is_jsonl;
    let before = score.then(|| sft::evaluate_word_task(&model, &tok, val_ex, eval_n, temperature, seed + 1));
    if let Some(e) = &before {
        println!("\nbefore fine-tuning:");
        print_eval("base", e);
    }

    let cfg = SftConfig { batch: args.get("batch", 16)?, base_lr: args.get("lr", 1e-3)?, weight_decay: 0.01, warmup: (steps / 20).clamp(1, 100), clip: 1.0, threads };
    let mut trainer = SftTrainer::new(model, cfg, train, val, steps, seed)?;
    println!("\nfine-tuning for {steps} steps (loss counts only the response tokens)...");
    println!("step {:>5}  response val loss {:.3}", 0, trainer.validate());
    let start = Instant::now();
    while trainer.step < steps {
        let loss = trainer.train_step();
        if trainer.step % 250 == 0 || trainer.step == steps {
            println!(
                "step {:>5}  train {loss:.3}  response val loss {:.3}  [{:.0}s]",
                trainer.step, trainer.validate(), start.elapsed().as_secs_f32()
            );
        }
    }

    // Save: weights, the tokenizer, and the chat template.
    let model = &trainer.model;
    checkpoint::save_model(out_dir, model, base_step + steps).map_err(|e| io_err("saving model", e))?;
    checkpoint::save_tokenizer(out_dir, &tok).map_err(|e| io_err("saving tokenizer", e))?;
    let after = score.then(|| sft::evaluate_word_task(model, &tok, val_ex, eval_n, temperature, seed + 1));
    let meta = SftMeta {
        base,
        task: if is_jsonl { "jsonl".into() } else { "use-the-word".into() },
        template: if is_jsonl { "{}".into() } else { sft::WORD_TEMPLATE.into() },
        stop: sft::STOP.into(),
        steps,
        train_examples: train_ex.len(),
        word_rate_before: before.as_ref().map(|e| e.uses_word),
        word_rate_after: after.as_ref().map(|e| e.uses_word),
        data_path: (!is_jsonl).then(|| data.clone()),
        text_start: (!is_jsonl).then_some(text_start),
        seed: Some(seed),
        max_examples: Some(max_examples),
        dpo: None,
    };
    std::fs::write(out_dir.join("sft.json"), serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?)
        .map_err(|e| io_err("saving sft.json", e))?;

    if let (Some(b), Some(a)) = (&before, &after) {
        println!("\nresults on {} held-out questions (temperature {temperature}):", a.n);
        print_eval("before", b);
        print_eval("after", a);
        println!("\nsample answers after fine-tuning:");
        for (prompt, text) in &a.samples {
            println!("  {:<24} ->{}", prompt.replace('\n', " "), text);
        }
    }
    println!("\nsaved to {}. Try it:  tlm chat --model {}", out_dir.display(), out_dir.display());
    Ok(())
}

fn cmd_dpo(raw: &[String]) -> Result<()> {
    let args = Args::parse(raw, &[
        "model", "out", "data", "prompts", "k", "steps", "lr", "beta", "sft-weight", "batch", "eval-n", "temperature", "mine-temperature", "threads", "seed",
    ])?;
    let (model_name, out) = (args.require("model")?, args.require("out")?);
    let (model_dir, out_dir) = (Path::new(&model_name), Path::new(&out));
    if out_dir.join("sft.json").exists() {
        return Err(format!("{out} already holds a model; choose a new --out"));
    }
    let meta_text = std::fs::read_to_string(model_dir.join("sft.json"))
        .map_err(|_| format!("{model_name} has no sft.json: DPO starts from a fine-tuned model. Run `tlm sft` first."))?;
    let meta: SftMeta = serde_json::from_str(&meta_text).map_err(|e| format!("bad sft.json: {e}"))?;
    if meta.task != "use-the-word" {
        return Err("DPO here is built for the \"use-the-word\" task (it needs a way to check answers); this model was fine-tuned on a custom dataset".into());
    }
    let data = match args.maybe("data") {
        Some(d) => d.clone(),
        None => meta.data_path.clone().ok_or("this sft.json does not record its data file; pass --data")?,
    };
    let steps: usize = args.get("steps", 600)?;
    let lr: f32 = args.get("lr", 1e-4)?;
    let beta: f32 = args.get("beta", 0.3)?;
    // Plain DPO (0) can lower the likelihood of the chosen answers too and wreck quality; see the dpo module.
    let sft_weight: f32 = args.get("sft-weight", 4.0)?;
    let k: usize = args.get("k", 4)?;
    let n_prompts: usize = args.get("prompts", 20_000)?;
    let batch: usize = args.get("batch", 16)?;
    let eval_n: usize = args.get("eval-n", 500)?;
    let temperature: f32 = args.get("temperature", 0.7)?;
    // Mine at the temperature the model is used at, so rejected answers are fluent but off-task (not gibberish).
    let mine_temperature: f32 = args.get("mine-temperature", temperature)?;
    let threads = args.get("threads", default_threads())?;
    let seed: u64 = args.get("seed", 11)?;
    if k < 2 || beta <= 0.0 || lr <= 0.0 || steps == 0 || batch == 0 || temperature <= 0.0 || mine_temperature <= 0.0 {
        return Err("need --k >= 2 and positive --beta, --lr, --steps, --batch and temperatures".into());
    }

    let (model, base_step) = checkpoint::load_model(model_dir).map_err(|e| io_err("loading model", e))?;
    let tok = checkpoint::load_tokenizer(model_dir).map_err(|e| io_err("loading tokenizer", e))?;

    // Rebuild exactly the examples (and the held-out split) that fine-tuning used.
    let data_seed = meta.seed.unwrap_or(7);
    let start = match meta.text_start {
        Some(offset) => StartAt::Offset(offset),
        None => StartAt::AfterBaseRun(Path::new(&meta.base)),
    };
    let (examples, _) = build_word_dataset(&data, start, meta.max_examples.unwrap_or(20_000), data_seed)?;
    if examples.len() < 100 {
        return Err(format!("only {} usable examples; need at least 100", examples.len()));
    }
    let (train_ex, val_ex) = examples.split_at(examples.len() - held_out_count(examples.len()));
    println!("{} prompts available for mining; {} held-out questions for scoring (never used for training)", train_ex.len(), val_ex.len());

    // Score the starting model on the held-out questions.
    let before = sft::evaluate_word_task(&model, &tok, val_ex, eval_n, temperature, data_seed + 1);
    println!("\nbefore preference training:");
    print_eval("sft", &before);

    // 1. Mine preference pairs from the model's own answers.
    let prompts: Vec<String> = train_ex.iter().take(n_prompts).map(|e| e.prompt.clone()).collect();
    println!("\nsampling {k} answers for each of {} prompts (temperature {mine_temperature}) and checking them...", prompts.len());
    let t0 = Instant::now();
    let (raw_pairs, stats) = dpo::mine_pairs(&model, &tok, &prompts, 0, k, mine_temperature, 60, seed, threads, &dpo::uses_word_and_stops);
    println!(
        "  {} answers sampled in {:.0}s; {:.1}% passed the check (uses the word and finishes properly)",
        stats.samples, t0.elapsed().as_secs_f32(), 100.0 * stats.good as f32 / stats.samples.max(1) as f32
    );
    println!("  {} prompts gave both a good and a bad answer, i.e. a preference pair ({:.0}% of prompts)", stats.pairs, 100.0 * stats.pairs as f32 / stats.prompts.max(1) as f32);
    if raw_pairs.len() < 50 {
        return Err("fewer than 50 preference pairs: the model answers almost everything the same way. Try more --prompts, a larger --k, or a higher --mine-temperature.".into());
    }
    let text_of = |ids: &[usize]| {
        let full = tok.decode(ids);
        full.find(sft::STOP).map_or(full.clone(), |at| full[..at].to_string())
    };
    println!("\nexample pairs:");
    for p in raw_pairs.iter().take(3) {
        println!("  {}", tok.decode(&p.prompt_ids).replace('\n', " "));
        println!("    chosen:   {}", text_of(&p.chosen).trim());
        println!("    rejected: {}", text_of(&p.rejected).trim());
    }

    // 2. Score the pairs with the frozen reference (the model as it is now), then train.
    let pairs = dpo::build_pairs(&model, &raw_pairs, threads);
    let n_val = (pairs.len() / 10).max(10).min(pairs.len() / 2);
    let (train_pairs, val_pairs) = pairs.split_at(pairs.len() - n_val);
    let cfg = DpoConfig { batch, base_lr: lr, beta, sft_weight, weight_decay: 0.0, warmup: (steps / 20).clamp(1, 50), clip: 1.0, threads };
    let mut trainer = DpoTrainer::new(model, cfg, train_pairs.to_vec(), val_pairs.to_vec(), steps, seed)?;
    let v0 = trainer.validate();
    println!("\npreference training: {} pairs ({} held out), beta {beta}, lr {lr}, {steps} steps", train_pairs.len(), val_pairs.len());
    println!("  loss starts at ln 2 = 0.693 and accuracy at 0%: the model has not moved away from the reference yet");
    println!("step {:>5}  val loss {:.3}  prefers chosen {:>5.1}%  margin {:+.2}", 0, v0.loss, v0.accuracy * 100.0, v0.margin);
    let start_time = Instant::now();
    let mut recent: Vec<dpo::DpoStats> = Vec::new();
    while trainer.step < steps {
        recent.push(trainer.train_step());
        if trainer.step % 50 == 0 || trainer.step == steps {
            let n = recent.len() as f32;
            let (l, a, m) = (recent.iter().map(|s| s.loss).sum::<f32>() / n, recent.iter().map(|s| s.accuracy).sum::<f32>() / n, recent.iter().map(|s| s.margin).sum::<f32>() / n);
            recent.clear();
            let v = trainer.validate();
            println!(
                "step {:>5}  train loss {l:.3} acc {:>4.0}% margin {m:+.2}   val loss {:.3}  prefers chosen {:>5.1}%  margin {:+.2}  [{:.0}s]",
                trainer.step, a * 100.0, v.loss, v.accuracy * 100.0, v.margin, start_time.elapsed().as_secs_f32()
            );
        }
    }

    // 3. Score again on the same held-out questions, and save.
    let model = &trainer.model;
    let after = sft::evaluate_word_task(model, &tok, val_ex, eval_n, temperature, data_seed + 1);
    checkpoint::save_model(out_dir, model, base_step + steps).map_err(|e| io_err("saving model", e))?;
    checkpoint::save_tokenizer(out_dir, &tok).map_err(|e| io_err("saving tokenizer", e))?;
    let new_meta = SftMeta {
        base: model_name.clone(),
        word_rate_after: Some(after.uses_word),
        dpo: Some(DpoMeta { beta, steps, pairs: pairs.len(), lr, word_rate_before: before.uses_word, word_rate_after: after.uses_word }),
        ..meta
    };
    std::fs::write(out_dir.join("sft.json"), serde_json::to_string_pretty(&new_meta).map_err(|e| e.to_string())?).map_err(|e| io_err("saving sft.json", e))?;

    println!("\nresults on {} held-out questions (temperature {temperature}):", after.n);
    print_eval("before", &before);
    print_eval("after", &after);
    println!("\nsample answers after preference training:");
    for (prompt, text) in &after.samples {
        println!("  {:<24} ->{}", prompt.replace('\n', " "), text);
    }
    println!("\nsaved to {}. Try it:  tlm chat --model {}", out_dir.display(), out_dir.display());
    Ok(())
}

fn cmd_chat(raw: &[String]) -> Result<()> {
    let args = Args::parse(raw, &["model", "input", "temperature", "tokens", "seed", "n"])?;
    let dir = args.require("model")?;
    let dir = Path::new(&dir);
    let meta_text = std::fs::read_to_string(dir.join("sft.json"))
        .map_err(|_| "this model has no sft.json: it is a base model. Fine-tune it with `tlm sft`, or use `tlm generate` to continue text.".to_string())?;
    let meta: SftMeta = serde_json::from_str(&meta_text).map_err(|e| format!("bad sft.json: {e}"))?;
    let (model, _) = checkpoint::load_model(dir).map_err(|e| io_err("loading model", e))?;
    let tok = checkpoint::load_tokenizer(dir).map_err(|e| io_err("loading tokenizer", e))?;
    let temperature: f32 = args.get("temperature", 0.7)?;
    let max_tokens: usize = args.get("tokens", 60)?;
    let mut rng = Rng::new(args.get("seed", 99)?);
    let n: usize = args.get("n", 1)?;
    if temperature <= 0.0 {
        return Err("--temperature must be greater than 0".into());
    }

    let mut respond = |input: &str| {
        let prompt = meta.template.replace("{}", input.trim());
        for _ in 0..n {
            let (text, stopped) = sft::answer(&model, &tok, &prompt, max_tokens, temperature, &mut rng);
            println!("{}{}", text.trim_start(), if stopped { "" } else { "  [no stop: hit the length limit]" });
        }
    };
    if let Some(input) = args.maybe("input") {
        respond(input);
        return Ok(());
    }
    println!("Fine-tuned for the \"{}\" task. Template: {:?}", meta.task, meta.template);
    println!("Type your input and press Enter (empty line to quit).\n");
    let stdin = std::io::stdin();
    loop {
        print!("> ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            break;
        }
        respond(line);
        println!();
    }
    Ok(())
}

const USAGE: &str = "\
tlm: tiny language models on the CPU, no GPU needed

USAGE:
  tlm train    --data <text file> --out <run dir> [--steps 3000] [--vocab 512]
               [--d-model 64] [--layers 2] [--heads 4] [--ctx 32] [--batch 16]
               [--lr 0.003] [--bytes 4000000] [--threads N] [--seed 42]
               [--eval-every 250] [--save-every 500]
  tlm resume   --out <run dir> [--steps <new total>] [--threads N]
  tlm generate --model <run dir> [--prompt \"text\"] [--tokens 100] [--temperature 0.8]
  tlm info     --model <run dir>
  tlm sft      --base <run dir> --out <new dir> --data <stories .txt | examples .jsonl>
               [--steps 1500] [--lr 0.001] [--batch 16] [--examples 20000] [--eval-n 100]
  tlm dpo      --model <fine-tuned dir> --out <new dir> [--data <stories .txt>]
               [--prompts 20000] [--k 4] [--steps 600] [--lr 0.0001] [--beta 0.3]
               [--sft-weight 4] [--mine-temperature 0.7] [--eval-n 500]
  tlm chat     --model <fine-tuned dir> [--input \"ball\"] [--n 3] [--temperature 0.7]";

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let result = match raw.first().map(String::as_str) {
        Some("train") => cmd_train(&raw[1..]),
        Some("resume") => cmd_resume(&raw[1..]),
        Some("generate") => cmd_generate(&raw[1..]),
        Some("info") => cmd_info(&raw[1..]),
        Some("sft") => cmd_sft(&raw[1..]),
        Some("dpo") => cmd_dpo(&raw[1..]),
        Some("chat") => cmd_chat(&raw[1..]),
        _ => {
            println!("{USAGE}");
            return;
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
