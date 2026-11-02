// A worker. Each page runs one LEADER and zero or more HELPERS, all identical copies
// of the same model (the leader talks to the page; helpers talk to the leader).
//
// With helpers, every training step is data-parallel:
//   1. each worker computes gradients for its share of the batch,
//   2. the leader adds the helpers' gradients to its own,
//   3. the leader sends the total back, and everyone applies the same update.
// Only gradients travel between workers, so the copies stay identical.

import init, { Session } from "./pkg/tlm_wasm.js";

let role = "leader";
let session = null;
let wasmReady = null;

// --- leader state ---
let helpers = []; // { port, pending: Map<id, {resolve, reject}> }
let nextRequestId = 1;
let running = false;
let loopActive = false;
let ema = null;          // smoothed training loss (raw batch loss is noisy)
let stepMs = 20;         // running estimate of milliseconds per step (single-worker mode)
let nextEvalStep = 0;    // evaluate / sample when training reaches these steps; the gaps adapt
let nextSampleStep = 0;  // to how long evaluating takes, so it never dominates training
let minEvalGap = 5;
let sampleGap = 50;
let lastProgressPost = 0;
let started = 0;
let stepsAtStart = 0;
let sampleInput = { prompt: "Once upon a time", word: "ball" };

// --- helper state ---
let leaderPort = null;

const post = (msg, transfer) => postMessage(msg, transfer);
const ensureWasm = () => (wasmReady ??= init());

self.onmessage = async (event) => {
  const m = event.data;
  try {
    switch (m.type) {
      case "attach": return attach(m);
      case "helperInit": return helperInit(m);
      case "init": return await onInit(m);
      case "restore": return await onRestore(m);
      case "loadModel": return await onLoadModel(m);
      case "startFinetune": return await onStartFinetune(m);
      case "startDpo": return await onStartDpo(m);
      case "start": return start(m);
      case "pause": running = false; return;
      case "generate": return onGenerate(m);
      case "answer": return onAnswer(m);
      case "saveState": return onSaveState();
      case "export": return onExport();
    }
  } catch (err) {
    running = false;
    post({ type: role === "helper" ? "helperError" : "error", message: String(err?.message ?? err) });
  }
};

// ---------- setting up the pool ----------

function attach({ port }) {
  role = "leader";
  const helper = { port, pending: new Map() };
  port.onmessage = (e) => {
    const { id, error, ...rest } = e.data;
    const waiting = helper.pending.get(id);
    if (!waiting) return;
    helper.pending.delete(id);
    error ? waiting.reject(new Error(`helper failed: ${error}`)) : waiting.resolve(rest);
  };
  helpers.push(helper);
}

function request(helper, msg) {
  return new Promise((resolve, reject) => {
    const id = nextRequestId++;
    helper.pending.set(id, { resolve, reject });
    helper.port.postMessage({ ...msg, id });
  });
}

function helperInit({ port }) {
  role = "helper";
  leaderPort = port;
  port.onmessage = async (e) => {
    const m = e.data;
    try {
      switch (m.type) {
        case "compute": {
          const grads = session.compute_grads(m.shard, m.n);
          leaderPort.postMessage({ id: m.id, grads }, [grads.buffer]);
          break;
        }
        case "apply":
          session.apply_grads(m.grads); // no reply: messages on one port arrive in order
          break;
        case "checksum":
          leaderPort.postMessage({ id: m.id, checksum: session.checksum() });
          break;
        case "mine": {
          const blob = session.mine_shard(m.settings, m.shard, m.n);
          leaderPort.postMessage({ id: m.id, blob }, [blob.buffer]);
          break;
        }
        case "setPairs":
          // Every worker scores the SAME merged pairs with the same frozen reference, so the copies stay identical.
          for (const blob of m.blobs) session.add_mined(blob);
          session.start_dpo(m.settings);
          leaderPort.postMessage({ id: m.id, ok: true });
          break;
      }
    } catch (err) {
      leaderPort.postMessage({ id: m.id, error: String(err?.message ?? err) });
    }
  };
}

// ---------- creating / restoring / loading a session ----------

function setSession(s) {
  session?.free();
  session = s;
  resetCounters();
}

/** Forget progress bookkeeping (a new session, or a new phase of the same one). */
function resetCounters() {
  ema = null;
  stepMs = 20;
  running = false;
  const total = session.total_steps();
  const now = session.step_count();
  minEvalGap = Math.max(5, Math.round(total / 150)); // at most ~150 chart points...
  sampleGap = Math.max(50, Math.round(total / 8)); // ...and ~8 text samples (more if they are cheap)
  nextEvalStep = now + minEvalGap;
  nextSampleStep = now + sampleGap;
}

function announceReady(extra = {}) {
  if (role === "helper") return post({ type: "helperReady" });
  const meta = JSON.parse(session.export_meta());
  post({ type: "ready", meta, valBpb: meta.phase === "pretrain" ? session.bits_per_byte(session.validate()) : null, ...extra });
}

async function onInit({ text, config }) {
  await ensureWasm();
  post({ type: "status", message: "Training the tokenizer and building the model…" });
  const t0 = performance.now();
  setSession(new Session(text, JSON.stringify(config)));
  announceReady({ setupMs: performance.now() - t0 });
}

async function onRestore({ text, meta, model, optimizer, tokenizer, pairs }) {
  await ensureWasm();
  post({ type: "status", message: "Restoring your saved session…" });
  const t0 = performance.now();
  setSession(Session.restore(text, meta, model, optimizer, tokenizer, pairs ?? new Uint8Array(0)));
  announceReady({ setupMs: performance.now() - t0 });
}

async function onLoadModel({ model, tokenizer, template }) {
  await ensureWasm();
  setSession(Session.from_model(model, tokenizer, template ?? undefined));
  announceReady();
}

async function onStartFinetune({ settings }) {
  if (role === "leader") post({ type: "status", message: "Building fine-tuning examples…" });
  session.start_finetune(JSON.stringify(settings));
  resetCounters(); // new phase, same session: restart the progress bookkeeping
  if (role === "helper") return post({ type: "helperReady" });
  post({ type: "status", message: "Scoring the model before fine-tuning…" });
  const before = JSON.parse(session.eval_word(100, 0.7, 1));
  post({ type: "finetuneReady", meta: JSON.parse(session.export_meta()), before, valLoss: session.validate() });
}

/** Preference tuning: mine pairs across all workers, score them, switch phase. Runs on the leader only. */
async function onStartDpo({ settings }) {
  const json = JSON.stringify(settings);
  const n = helpers.length + 1;
  post({ type: "status", message: "Scoring the fine-tuned model before preference tuning…" });
  const before = JSON.parse(session.eval_word(200, 0.7, 1));

  post({ type: "status", message: `Sampling ${settings.k} answers for each of ${settings.prompts} prompts${n > 1 ? ` (${n} workers)` : ""}…` });
  const waiting = helpers.map((h, i) => request(h, { type: "mine", settings: json, shard: i + 1, n }));
  const own = session.mine_shard(json, 0, n);
  const blobs = [own, ...(await Promise.all(waiting)).map((r) => r.blob)]; // in shard order: joined, they equal mining everything at once

  post({ type: "status", message: "Scoring the pairs with the frozen reference model…" });
  // The leader goes first: if there are too few pairs it fails here, before any helper has switched phase.
  for (const blob of blobs) session.add_mined(blob);
  session.start_dpo(json);
  await Promise.all(helpers.map((h) => request(h, { type: "setPairs", blobs, settings: json })));
  resetCounters();
  post({ type: "dpoReady", meta: JSON.parse(session.export_meta()), before, valLoss: session.validate(), accuracy: session.preference_accuracy() });
}

// ---------- training (leader) ----------

function start(msg) {
  if (role !== "leader" || !session) throw new Error("Prepare a model first.");
  sampleInput = { prompt: msg.prompt || sampleInput.prompt, word: msg.word || sampleInput.word };
  if (loopActive) return;
  running = true;
  started = performance.now();
  stepsAtStart = session.step_count();
  trainLoop();
}

async function trainLoop() {
  loopActive = true;
  let finished = false;
  try {
    while (running && !finished) finished = await iterate();
  } catch (err) {
    running = false;
    post({ type: "error", message: String(err?.message ?? err) });
  }
  loopActive = false;
  if (finished) return finish();
  if (!running) post({ type: "paused", step: session.step_count(), phase: session.phase() });
}

/** One unit of training work. Returns true when the phase has finished. */
async function iterate() {
  const total = session.total_steps();
  if (session.step_count() >= total) return true;

  const before = session.step_count();
  const t0 = performance.now();
  let loss;
  if (helpers.length === 0) {
    // Aim for ~40 ms per chunk: long enough to be efficient, short enough that
    // pause / generate messages are noticed quickly.
    const chunk = Math.max(1, Math.min(50, Math.round(40 / stepMs)));
    loss = session.step(chunk);
  } else {
    loss = await parallelStep();
  }
  const done = session.step_count();
  stepMs = 0.7 * stepMs + 0.3 * ((performance.now() - t0) / Math.max(1, done - before));
  ema = ema === null ? loss : 0.9 * ema + 0.1 * loss;
  report(done, total);

  // Yield so incoming messages are handled. Parallel mode already yields while
  // awaiting the helpers; waiting on a timer there would only add delay.
  if (helpers.length === 0) await new Promise((r) => setTimeout(r, 0));
  return done >= total;
}

// Where the leader's time goes in a parallel step (milliseconds, summed over steps).
const spent = { own: 0, wait: 0, add: 0, send: 0, apply: 0, steps: 0 };

async function parallelStep() {
  const n = helpers.length + 1;
  const t0 = performance.now();
  const waiting = helpers.map((h, i) => request(h, { type: "compute", shard: i + 1, n }));
  const mine = session.compute_grads(0, n); // flat gradients; the last number is this share's loss
  const t1 = performance.now();
  const theirs = await Promise.all(waiting); // time spent here = waiting for the slowest helper
  const t2 = performance.now();
  for (const { grads } of theirs) for (let i = 0; i < mine.length; i++) mine[i] += grads[i];
  const t3 = performance.now();
  for (const h of helpers) h.port.postMessage({ type: "apply", grads: mine }); // copied to each helper
  const t4 = performance.now();
  session.apply_grads(mine);
  const t5 = performance.now();
  spent.own += t1 - t0; spent.wait += t2 - t1; spent.add += t3 - t2; spent.send += t4 - t3; spent.apply += t5 - t4; spent.steps++;
  return mine[mine.length - 1];
}

/** Average milliseconds per parallel step, by phase of the step. */
function breakdown() {
  if (!spent.steps) return null;
  const per = (v) => +(v / spent.steps).toFixed(1);
  return { own_compute: per(spent.own), wait_for_helpers: per(spent.wait), add_gradients: per(spent.add), send_total: per(spent.send), apply_update: per(spent.apply) };
}

function report(done, total) {
  const phase = session.phase();
  const now = performance.now();
  const doEval = done >= nextEvalStep || done >= total;
  if (!doEval && now - lastProgressPost < 60) return; // at most ~16 updates a second
  lastProgressPost = now;

  const progress = {
    type: "progress",
    phase,
    step: done,
    total,
    train: phase === "pretrain" ? session.bits_per_byte(ema) : ema,
    stepsPerSec: ((done - stepsAtStart) / (now - started)) * 1000,
    workers: helpers.length + 1,
  };
  if (doEval) {
    const t0 = performance.now();
    const v = session.validate();
    progress.val = phase === "pretrain" ? session.bits_per_byte(v) : v;
    if (phase === "dpo") progress.valAcc = session.preference_accuracy();
    // Evaluation runs on the leader alone (helpers wait), so keep it to ~10% of the time.
    nextEvalStep = done + Math.max(minEvalGap, Math.ceil((10 * (performance.now() - t0)) / Math.max(stepMs, 1)));
  }
  post(progress);

  if (done >= nextSampleStep || done >= total) {
    const t0 = performance.now();
    const text =
      phase !== "pretrain"
        ? `${sampleInput.word} → ${session.answer(sampleInput.word, 60, 0.7, done)}`
        : sampleInput.prompt + session.generate(sampleInput.prompt, 60, 0.8, done);
    post({ type: "sample", phase, step: done, text });
    nextSampleStep = done + Math.max(sampleGap, Math.ceil((10 * (performance.now() - t0)) / Math.max(stepMs, 1)));
  }
}

async function finish() {
  const phase = session.phase();
  // Check that every copy of the model is still identical.
  let syncOk = true;
  if (helpers.length) {
    const mine = session.checksum();
    const others = await Promise.all(helpers.map((h) => request(h, { type: "checksum" })));
    syncOk = others.every((o) => o.checksum === mine);
  }
  let after = null;
  if (phase === "finetune" || phase === "dpo") {
    post({ type: "status", message: "Scoring the model…" });
    after = JSON.parse(session.eval_word(200, 0.7, 1));
  }
  running = false;
  post({ type: "done", phase, step: session.step_count(), syncOk, after, workers: helpers.length + 1, breakdown: breakdown() });
}

// ---------- everything else the page can ask the leader ----------

function onGenerate({ id, prompt, n, temperature, seed }) {
  post({ type: "generated", id, text: prompt + session.generate(prompt, n, temperature, seed) });
}

function onAnswer({ id, input, temperature, seed }) {
  const texts = [0, 1, 2].map((i) => session.answer(input, 80, temperature, seed + i));
  post({ type: "answered", id, texts });
}

function onSaveState() {
  const phase = session.phase();
  if (phase === "loaded") return post({ type: "snapshot", unavailable: true });
  const model = session.export_model();
  const optimizer = session.export_optimizer();
  const pairs = session.export_pairs(); // empty unless in preference tuning
  post(
    { type: "snapshot", meta: session.export_meta(), model, optimizer, tokenizer: session.export_tokenizer(), pairs },
    [model.buffer, optimizer.buffer, pairs.buffer]
  );
}

function onExport() {
  const model = session.export_model();
  post({ type: "exported", model, tokenizer: session.export_tokenizer(), meta: JSON.parse(session.export_meta()) }, [model.buffer]);
}
