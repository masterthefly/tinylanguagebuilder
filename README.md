# TinyLanguageBuilder

Build, train and understand a tiny language model on your own CPU: no GPU, no ML framework, no account, no server.
It runs natively as a command-line tool (`tlm`) and **in a browser tab** (using WebAssembly and Web Workers), and it
comes with an **interactive textbook** that opens up the same engine one idea at a time.

Everything is written from scratch in Rust, including matrix multiplication, backpropagation, attention, the
AdamW optimizer, and a byte-level BPE tokenizer. It is a teaching project: the code is meant to be read.

It is inspired by [languagemodelbuilder.com](https://languagemodelbuilder.com/) by Felix Rieseberg, a macOS app
with an interactive textbook and a workbench for pre-training, fine-tuning and DPO. This project runs anywhere.

## Start here

- **Learn:** open the interactive textbook (`web/textbook/`, linked from the app). Five short chapters (tokens,
  embeddings, attention, learning, sampling) where every widget runs the real engine in your browser. Select any
  text and press **Explain** for a plain-language definition.
- **Build:** open the app, pick a model size, and train a model in your browser tab. Fine-tune it and chat with it.
- **Use it on your own material:** train on any `.txt` file, or teach it your own examples from a `.jsonl` file.

## What it can do

- **Pre-train** a GPT-style transformer on plain text (tested on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)).
- **Fine-tune (SFT)** it on a verifiable task ("write a sentence using this word") or on **your own examples**
  (a `.jsonl` file of `{"prompt": ..., "response": ...}` lines), in the browser or with `tlm sft`.
- **Preference-tune (DPO)** it, mining preference pairs from its own samples.
- **Chat** with the result, save and resume sessions, and open model files (safetensors).
- Train in parallel across Web Workers in the browser, or across threads natively.

## Repository layout

```
crates/
  tlm-core/    the engine: matrix, ops, BPE, attention, transformer, optimizer, SFT, DPO, checkpoints
  tlm-cli/     the `tlm` binary: train / resume / generate / info / sft / dpo / chat
  tlm-wasm/    wasm-bindgen bindings used by the web app
web/           the browser app (index.html, app.js, worker.js, store.js, style.css)
scripts/       web build script, browser end-to-end test, benchmarks
```

## Requirements

- [Rust](https://rustup.rs/) (stable) with the wasm target: `rustup target add wasm32-unknown-unknown`
- For the browser app: `wasm-bindgen-cli` **0.2.130** (it must match the version pinned in
  `crates/tlm-wasm/Cargo.toml`): `cargo install wasm-bindgen-cli --version 0.2.130`
- Python 3 (only to serve the web folder statically)
- Node.js (only for the browser end-to-end test)

## Get training data

The data is not stored in the repository. Download the TinyStories validation file (about 22 MB):

```powershell
New-Item -ItemType Directory -Force data/tinystories | Out-Null
curl.exe -L -o data/tinystories/valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

## Command-line quick start

```powershell
cargo test --workspace         # all native tests (about 15 s)
cargo build --release          # builds target\release\tlm.exe

# 1. pre-train a small model
.\target\release\tlm.exe train --data data/tinystories/valid.txt --out runs/base64 --ctx 64 --steps 4000

# 2. teach it the "use the word" task
.\target\release\tlm.exe sft --base runs/base64 --out runs/sft-small --data data/tinystories/valid.txt --steps 5000 --eval-n 1000

# 3. preference-tune it
.\target\release\tlm.exe dpo --model runs/sft-small --out runs/dpo-small --data data/tinystories/valid.txt --eval-n 1000

# 4. talk to it
.\target\release\tlm.exe chat --model runs/dpo-small --input garden --n 3
```

On Linux or macOS, use `./target/release/tlm` and forward slashes. Run `tlm --help` for all options.

## Browser app quick start

The app needs a sample of the training text at `web/data/tinystories-8mb.txt`: the first 8 MB of
`valid.txt`, cut after the last `<|endoftext|>`. `PROJECT_STATUS.md` (section 8) has a PowerShell snippet that
creates it.

```powershell
.\scripts\build-web.ps1                          # compiles to WebAssembly into web/pkg
python -m http.server 8080 --directory web       # then open http://localhost:8080/
```

Pick a model size (Tiny, Small or Medium), pre-train, fine-tune, preference-tune, and chat. Sessions autosave
to IndexedDB and can be resumed or downloaded as model files.

## Model sizes

| Name | d_model / layers / heads / vocab | Parameters |
|---|---|---|
| Tiny | 32 / 2 / 2 / 512 | ~59k |
| Small (default) | 64 / 2 / 4 / 512 | ~169k |
| Medium | 128 / 4 / 4 / 1024 | ~1.06M |

The context window is 64 tokens.

## Results

Measured on this project's own experiments. The "uses the word" figures come from 1,000 held-out prompts, each
reported with a control (the same answers checked against a different question's word).

| Stage | Small | Medium |
|---|---|---|
| Pre-training (bits per byte, lower is better) | 1.37 | 0.94 |
| After SFT: uses the requested word | 66.8% (control 8.9%) | 86.2% (control 7.2%) |
| After DPO + supervised term | 77.4% | 90.1% |

Training-step speed: 28 ms to 7.4 ms per step from the AVX2/FMA matmul kernel and other optimizations.
WebAssembly SIMD is 5.7x faster than scalar wasm, and 8 browser workers give about 4x on the Small model.

Honest caveats: most of the DPO gain comes from its supervised term, not from the preference part. The DPO
settings were compared on the same held-out questions, so 77.4% is slightly optimistic. Single runs carry about
±1.5 points of noise.

## Correctness

Every gradient (embeddings, LayerNorm, attention, GELU, masked loss, the DPO loss) is checked numerically
against finite differences. There are 130 native tests plus browser end-to-end tests: 51 checks for the app, and one
script for each textbook chapter, for the "Explain" glossary and for fine-tuning on your own examples. They are plain
[Playwright](https://playwright.dev/) scripts. To run one:

```powershell
cd scripts; npm install; npx playwright install chromium; cd ..
node scripts/run-e2e.mjs http://localhost:8080/ scripts/e2e-web.mjs
```

## Limitations

- One automatically scored task ("use the word"). Your own examples can be fine-tuned on in the browser and the CLI, but
  they have no automatic checker, so there is no preference tuning (DPO) for them and the page reports exact matches and
  side-by-side answers instead. With only a few dozen examples the model memorises them (the page warns when held-out
  loss rises); rare words are hit-or-miss on the small model.
- Only Chromium has been tested. Firefox and Safari are untested.
- Browser scaling flattens past 4–8 workers because all gradients pass through one leader worker.
- Results are exact on one machine but not bit-identical across different CPUs.
- The `tlm` CLI has no unit tests of its own.

## License

MIT. See [LICENSE](LICENSE).

## More documentation

- `PROJECT_STATUS.md`: full status, lesson history, design notes, findings, and the prioritized roadmap.
