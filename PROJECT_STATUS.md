# TinyLanguageBuilder: project status

_Last updated: 2026-11-02. A snapshot of what is built, how to run it, what was measured and what is next._

## 1. What this project is

A **CPU-only (no GPU) tiny language model builder**, inspired by
[languagemodelbuilder.com](https://languagemodelbuilder.com/) (a free macOS-only app by Felix Rieseberg:
an interactive textbook plus a workbench for pre-training, supervised fine-tuning (SFT) and DPO).

Ours runs anywhere: natively (Windows/Linux/Mac, multi-threaded) and **in a browser tab** (WebAssembly in Web
Workers). Everything is written from scratch in Rust (no ML framework, no autograd), built up lesson by lesson
as a teaching project.

Two products share one core:

| Product | What it is |
|---|---|
| `tlm` (command-line tool) | train / resume / generate / info / sft / dpo / chat |
| TinyLanguageBuilder (`web/`) | a web page: pre-train, fine-tune, preference-tune, chat, save/resume, share models |

## 3. Where things stand

**Everything below works and is tested.** 130 native tests pass (96 in `tlm-core`, 34 in `tlm-wasm`), plus a
51-check end-to-end browser test for the app and one end-to-end script per textbook chapter.

### Lessons completed

| # | Topic | Key result |
|---|---|---|
| 0 | Setup (Rust, wasm target) | rustc 1.99.1 |
| 1 | `Matrix`, matmul | row-major, tested |
| 2 | Stable softmax | the overflow trap, demonstrated in a test |
| 3 | Bigram model, loss, gradient descent | first gradient check |
| 4 | Context + embeddings + MLP | showed overfitting (train vs validation) |
| 5 | Byte-level BPE + real data (TinyStories) | 2.4 bytes/token; MLP 2.02 bits/byte |
| 6 | Causal self-attention, forward+backward | gradient-checked |
| 7 | Full GPT-style transformer, AdamW, threads | 1.38 bits/byte in 25 s |
| 8 | Safetensors checkpoints, resume, `tlm` CLI | resume is bit-exact |
| 9 | Performance (AVX2/FMA register-blocked matmul, fast tanh, cached transposes) | step 28 ms -> 7.4 ms (3.8x) |
| 10 | WebAssembly + browser UI | SIMD gives 5.7x in wasm |
| 11 | SFT (loss masking, "use the word" task) | 4.8% -> 66.8% (control ~9%) |
| 11b | One batch of four upgrades: SFT in the browser, persistence (IndexedDB autosave/resume, open model files), bigger model, multi-worker training | ~4x on 8 workers |
| 12 | DPO (core + CLI), on-policy mining, ablations | see section 6 |
| 12b | DPO in the browser (parallel mining, mid-DPO resume) | 51/51 e2e checks |

### Architecture

```
tinylanguagemodel/
  Cargo.toml                    workspace; [profile.test] opt-level=2 (tests ~100x faster)
  .cargo/config.toml            wasm32 target: +simd128
  crates/
    tlm-core/                   the engine (no I/O beyond checkpoints); ~4,150 lines
      matrix.rs                 Matrix + fast matmul (AVX2/FMA kernel, runtime-detected; matmul_at_b)
      ops.rs, rng.rs, profile.rs  softmax, xorshift RNG, optional `--features profile` span timer
      bpe.rs                    byte-level BPE (train/encode/decode/json)
      attention.rs, transformer.rs  hand-written forward + backward; per-sequence scales; IGNORE_TARGET mask
      optim.rs, trainer.rs      AdamW + clipping; Trainer with compute_grads/apply_grads (data-parallel)
      parallel.rs               shard ranges, gradient flatten/sum, checksum
      checkpoint.rs             safetensors read/write (validated), model/optimizer/tokenizer, in-memory bytes
      sft.rs                    examples, packing with masks, word-task dataset, SftTrainer, evaluation + control
      dpo.rs                    pair mining (verifier), pair encode/decode, DpoTrainer, supervised term
      bigram.rs, mlp.rs         the early lessons' models (kept)
      examples/                 demos and benchmarks (bench.rs, bench_ops.rs, *_demo.rs)
    tlm-cli/                    `tlm` binary (src/main.rs ~640 lines)
    tlm-wasm/                   wasm-bindgen `Session` (pretrain/finetune/dpo/loaded phases) ~1,000 lines
  web/                          index.html, style.css, app.js (UI), worker.js (leader/helper), store.js (IndexedDB)
  scripts/                      build-web.ps1, e2e-web.mjs, bench-workers.mjs, bench-wasm.mjs
  PROJECT_STATUS.md             this file
```

Key design ideas worth remembering:

- **Every gradient is verified numerically** against finite differences (embeddings, LayerNorm, attention,
  GELU, masked loss, weighted/negative-weight loss, the true DPO loss). This is the safety net.
- **Data-parallel training** (browser Web Workers, no SharedArrayBuffer): all workers draw the *same* batch,
  each computes its share's gradients (normalized by the *whole* batch's token count), the leader sums them,
  everyone applies the same update. Tested bit-identical across workers.
- **Leader/helper workers** talk over `MessageChannel` ports; the page only talks to the leader.
  Mining for DPO is sharded the same way (each prompt seeded by its *global* index, so sharding is exact).
- **Phases** of a session: `pretrain` -> `finetune` -> `dpo`; `loaded` (open a model file, chat only).
- **Saving a session** stores model + optimizer + tokenizer + text + UI history (+ scored DPO pairs, because a
  pair's reference score cannot be recomputed after the model moves).
- **Evaluation discipline:** "uses the requested word" is always reported with a **control** (the same answers
  checked against *another* question's word) on held-out prompts built from text the base model never saw.
- Browser test drivers evaluate in an isolated JS world, so tests read state from `<body data-tlb>`.

## 4. How to run (cheat sheet)

Open PowerShell in the project folder. If `cargo` is not found in a new shell:
`$env:Path += ";$env:USERPROFILE\.cargo\bin"`.

```powershell
cd <the repository folder>

cargo test --workspace                      # all native tests (about 15 s)
cargo build --release                       # builds the `tlm` tool at target\release\tlm.exe

# native pipeline
.\target\release\tlm.exe train --data data/tinystories/valid.txt --out runs/base64 --ctx 64 --steps 4000
.\target\release\tlm.exe sft   --base runs/base64 --out runs/sft-small --data data/tinystories/valid.txt --steps 5000 --eval-n 1000
.\target\release\tlm.exe dpo   --model runs/sft-small --out runs/dpo-small --data data/tinystories/valid.txt --eval-n 1000
.\target\release\tlm.exe chat  --model runs/dpo-small --input garden --n 3

# browser app
.\scripts\build-web.ps1                     # compiles to WebAssembly + generates web/pkg
python -m http.server 8080 --directory web  # then open http://localhost:8080/  (stop it when done)

# browser end-to-end tests (need the server running; see the README for the one-time Playwright setup)
node scripts/run-e2e.mjs http://localhost:8080/ scripts/e2e-web.mjs
```

### Things that are NOT in the repo and must exist (all git-ignored)

| Path | How to recreate |
|---|---|
| `data/tinystories/valid.txt` (22 MB) | `curl.exe -L -o data/tinystories/valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt` |
| `web/data/tinystories-8mb.txt` | the first 8 MB of the file above, cut after the last `<|endoftext|>` (see the PowerShell snippet in section 8) |
| `web/pkg/` | `.\scripts\build-web.ps1` |
| `runs/` | local experiment outputs (models); safe to delete, recreate with the commands above |
| `target/` | build output |

**Installed tools:** Rust via rustup (+ target `wasm32-unknown-unknown`), `wasm-bindgen-cli` **0.2.130**
(must exactly match the `wasm-bindgen = "=0.2.130"` pin in `crates/tlm-wasm/Cargo.toml`), Python 3 (static
server), Node 24.

## 5. Model sizes used

| Name | d_model / layers / heads / vocab | Params | Notes |
|---|---|---|---|
| Tiny (browser) | 32 / 2 / 2 / 512 | ~59k | fastest |
| Small | 64 / 2 / 4 / 512 | ~169k | default; ctx 64 |
| Medium | 128 / 4 / 4 / 1024 | ~1.06M | best quality; slow in the browser |

Context window is 64 tokens (needed so an instruction + answer fit).

## 6. Results and findings (what the experiments taught us)

**Pre-training (bits per byte on held-out text; lower is better; 8 = clueless)**
bigram ~3.1 (2.13 nats/char) -> MLP 2.02 -> small transformer 1.37 -> **medium transformer 0.94** (12 MB text, 8,000 steps, ~4.5 min).

**Speed.** One-thread training step 28 ms -> 7.4 ms (matmul kernel, cached transposes, fast tanh).
WebAssembly SIMD: 5.7x faster than scalar wasm. Browser workers (steady state):

| Model | 1 worker | 2 | 4 | 8 |
|---|---|---|---|---|
| Small | 16 steps/s | 33 | 52 | **66 (4.1x)** |
| Medium | 3 | 6 | 9 | **10 (3.3x)** |

Limit: every gradient flows through the leader (`workers x params` numbers per step). Evaluation/sampling were
also throttled adaptively (<= ~10% of time); that alone doubled real speed.

**SFT ("write a sentence using this word"), 1,000 held-out questions**

| | Uses the word | Control | Ends properly |
|---|---|---|---|
| base model | ~5% | ~4% | ~13% |
| small SFT | 66.8% | 8.9% | 100% |
| medium SFT | **86.2%** | 7.2% | 100% |

**DPO (small model, from SFT 66.8%)**

| Setting | Uses the word |
|---|---|
| plain DPO (mined at temperature 1.0) | 64.0% (worse) |
| plain DPO, mined at 0.7, 800 steps | 54.8% (much worse; garbled text) |
| supervised term only (beta ~ 0), no negatives | 74.2% |
| **DPO + supervised term (defaults: beta 0.3, `--sft-weight 4`, mine at 0.7)** | **77.4%** |
| medium: DPO + supervised vs supervised only | 90.1% vs 89.7% (preference part ~ noise) |

Takeaways: (1) plain DPO caused **likelihood displacement** (chosen answers' probability also fell -> gibberish);
(2) mining at temperature 1.0 gave *gibberish* rejected answers, so the model learned "be fluent", not the task;
(3) **most of the gain came from the supervised term** (rejection-sampling fine-tuning); the genuine preference
part adds ~3 points on the small model and ~0 on the medium; (4) single runs, about +/-1.5 points of noise.

## 7. Known limitations and caveats

- **Not a git repository yet.** There is no version control and no backup of the source. Do this first.
- **One task only:** "use the word" (verifiable). Rare, user-chosen words are still hit-or-miss on the small model.
- **Tuned on the evaluation set:** ~10 DPO settings were compared on the same 1,000 held-out questions, so
  the 77.4% is slightly optimistic. A fresh split is needed for an honest number.
- **Browser DPO is noisy with small models:** after 300+300 steps only ~60 pairs come from 300 prompts, and it
  overfits (the chart shows train loss ~0.02 vs held-out ~0.29). Works better after a longer fine-tune.
- **Scaling saturates** past 4-8 workers (all-reduce through one leader).
- **Results are not bit-identical across CPUs** (FMA rounds once instead of twice); same machine is exact.
- **Only Chromium was tested** (headless). Firefox/Safari untested. The medium preset was not run through
  the browser DPO flow.
- Samples show `<|endoftext|>` literally and occasional `?` (partial UTF-8) early in training; expected.
- The CLI has no unit tests (only the library crates do); `tlm` is covered by manual runs and the browser test.

## 8. Handy snippets

Recreate the 8 MB browser sample (PowerShell):

```powershell
$root = (Get-Location).Path
$bytes = [IO.File]::ReadAllBytes("$root\data\tinystories\valid.txt")[0..7999999]
$text  = [Text.Encoding]::UTF8.GetString($bytes)
$cut   = $text.LastIndexOf("<|endoftext|>") + "<|endoftext|>".Length
New-Item -ItemType Directory -Force "$root\web\data" | Out-Null
[IO.File]::WriteAllBytes("$root\web\data\tinystories-8mb.txt", [Text.Encoding]::UTF8.GetBytes($text.Substring(0, $cut) + "`n"))
```

Stop a leftover web server (a server started with `Start-Process` is not stopped by killing the listener owner):

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "http.server 8080" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

## 9. Next steps

Done since the first version: version control, the README, the interactive textbook (five chapters in
`web/textbook/`, with a built-in glossary behind the "Explain" button), and fine-tuning on your own examples
(`.jsonl`) in the browser. Still open:

1. **GitHub Pages deploy.** Publish `web/` (needs the sample text and the built `pkg/`; a small CI workflow could
   build them). The page already works from any static host.
2. **A second automatically checkable task**, so preference tuning (DPO) is not limited to "use the word". Custom
   `.jsonl` examples have no checker, so DPO is disabled for them.
3. **Honest DPO error bars.** Repeat the DPO comparison over several seeds on a *fresh* held-out split; report
   mean +/- spread, including the supervised-only ablation.
4. **Scale/quality work (optional):** tree or ring all-reduce to scale past 4-8 workers; shared-memory wasm
   threads (needs nightly Rust + COOP/COEP headers); bigger context and more data; a larger default
   browser model.
5. **Polish:** test the medium preset through the browser DPO flow; try Firefox/Safari; a UI toggle for a light/dark
   theme; CLI unit tests; show a "workers" speed/efficiency breakdown (the leader already collects timings);
   make the browser's minimum custom-example count (20) match the command line's (50).
