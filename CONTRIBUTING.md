# Contributing to TinyLanguageBuilder

Thank you for wanting to help. This is a **teaching project**: a language model small enough to read, run and
change on your own CPU, plus an interactive textbook that explains it. Contributions that make it clearer, more
correct or more useful to a learner are all welcome, and you do not need to be a machine-learning expert.

Please be kind and patient with each other. Assume good intentions, explain your reasoning, and keep feedback about
the code, not the person.

## Ways to help

- **Report a bug or a confusing explanation.** A textbook sentence that misled you is as valuable as a crash.
- **Improve the textbook.** Clearer wording, a better example, a new widget, or an accessibility fix.
- **Add glossary entries** for the "Explain" popover (`web/textbook/glossary.js`).
- **Run and report experiments.** Results with honest error bars are welcome (see "Reporting results" below).
- **Fix something from the roadmap.** `PROJECT_STATUS.md` (section 9) lists open ideas, for example a second
  automatically checkable task for preference tuning, or making the browser's minimum number of custom examples
  match the command line's.
- **Try it in another browser or on another operating system.** Only Chromium on Windows has been tested so far.

If you are unsure whether an idea fits, open an issue first and describe what you would like to change and why.

## Set up

You need:

- [Rust](https://rustup.rs/) (stable) with the wasm target: `rustup target add wasm32-unknown-unknown`
- `wasm-bindgen-cli` **0.2.130** exactly (it must match the version pinned in `crates/tlm-wasm/Cargo.toml`):
  `cargo install wasm-bindgen-cli --version 0.2.130`
- Python 3 (only to serve the `web/` folder) and Node.js (for the sample-data script and the browser tests)

Then, from the repository folder:

```powershell
cargo test --workspace                 # all native tests; takes a few seconds once built

# the web app and textbook
curl.exe -L -o data/tinystories/valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
node scripts/make-sample.mjs           # builds web/data/tinystories-8mb.txt
.\scripts\build-web.ps1                # compiles the engine to WebAssembly into web/pkg
python -m http.server 8080 --directory web
```

(Create `data/tinystories` first if it does not exist. On Linux or macOS use the same commands without the
PowerShell-specific `.\` and `curl.exe`; the two `cargo build` and `wasm-bindgen` lines in
`.github/workflows/pages.yml` show what `build-web.ps1` does.)

Open <http://localhost:8080/> for the app and <http://localhost:8080/textbook/> for the textbook. Stop the server
when you are done.

## Where things live

| You want to change | Look in |
|---|---|
| The model, tokenizer, optimizer, fine-tuning, DPO | `crates/tlm-core/src/` |
| The command-line tool (`tlm`) | `crates/tlm-cli/src/main.rs` |
| What the browser can call (sessions, training, textbook widgets) | `crates/tlm-wasm/src/lib.rs`, `crates/tlm-wasm/src/textbook.rs` |
| The app page | `web/index.html`, `web/app.js`, `web/worker.js`, `web/style.css` |
| A textbook chapter | `web/textbook/chapterN.html` and `chapterN.js` |
| Definitions behind "Explain" | `web/textbook/glossary.js` (matching logic in `explain.js`) |
| End-to-end browser tests | `scripts/e2e-*.mjs` (run with `scripts/run-e2e.mjs`) |

## Making a change

1. **Keep it focused.** One idea per pull request is much easier to review than several.
2. **Write the test with the change.** Native tests live next to the code (`#[cfg(test)]`). If you add or change a
   piece of the model that has a gradient (an operation, a layer, a loss), check that gradient numerically against
   finite differences, as the existing tests do. A wrong gradient trains silently and badly.
3. **Keep the code readable.** This project is meant to be read, so prefer clear code over clever code, and write
   comments that explain *why*, not *what*.
4. **Match the surrounding style.** The Rust code is not formatted with `rustfmt`, so please do not run it over
   whole files; that would bury your change in unrelated edits. The web code is plain ES modules with no build step
   and no framework. Please keep it that way.
5. **Keep the project's promises.** No GPU and no ML framework. Everything runs on the CPU, in a browser tab or
   natively. Nothing the user types or trains on is sent to a server. Avoid adding dependencies; if one seems
   necessary, explain why in the issue first.
6. **Watch for warnings.** `cargo test --workspace` currently builds with none. Please keep it that way.

### Textbook changes

- Widgets should use the **real engine** (the WebAssembly build of `tlm-core`), not a mock-up. If a chapter needs
  something the engine does not expose yet, add a small function to `crates/tlm-wasm/src/textbook.rs` with a test.
- Say what a result does and does not show. The chapters end with "what to take from this, and what not to".
- Make it usable without a mouse and on a phone, and do not rely on colour alone to carry meaning. Look at it in
  the light and dark themes.
- Add a check for new behaviour to the chapter's script in `scripts/` (or add a new `e2e-*.mjs`).
- New technical terms belong in `web/textbook/glossary.js`. Write the definition by hand and keep it accurate for
  *this* project's model. A test checks the glossary's integrity.

### Browser tests

The end-to-end scripts are plain [Playwright](https://playwright.dev/) scripts. One-time setup, then run any of them:

```powershell
cd scripts; npm install; npx playwright install chromium; cd ..
node scripts/run-e2e.mjs http://localhost:8080/ scripts/e2e-web.mjs
node scripts/run-e2e.mjs http://localhost:8080/textbook/chapter3.html scripts/e2e-explain.mjs
```

Each prints its checks as JSON and exits with a non-zero status if any fail. A change to the app should keep
`e2e-web.mjs` (51 checks) and `e2e-custom.mjs` passing, and a change to a chapter should keep that chapter's
script passing. Look at the result with your own eyes as well: a passing check does not prove that a page
looks right.

## Reporting results

Training results are noisy: a single run carries about ±1.5 points of run-to-run variation on the "uses the word"
score. If you report an improvement, please say how many runs you made, on what data, and compare against a
baseline with the same settings. Negative and noisy results are useful, so please share them too.

## Pull requests

- Describe what changed and why, and how you tested it. The pull request template has a short checklist.
- Make sure `cargo test --workspace` passes and that you ran the relevant browser tests.
- Do not commit generated or downloaded files: `target/`, `data/`, `runs/`, `web/pkg/` and `web/data/` are
  git-ignored on purpose, and the Pages build recreates the last two.
- Small, clearly explained pull requests get reviewed fastest.

## License

By contributing you agree that your contribution is licensed under the project's [MIT License](LICENSE).
