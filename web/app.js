// The page: buttons, the worker pool, saving, and the loss chart.

import { saveSession, loadSession, deleteSession } from "./store.js";

const PRESETS = {
  tiny:   { steps: 2000, config: { d_model: 32,  n_layer: 2, n_head: 2, vocab_size: 512,  lr: 0.003, max_bytes: 1_000_000 } },
  small:  { steps: 3000, config: { d_model: 64,  n_layer: 2, n_head: 4, vocab_size: 512,  lr: 0.003, max_bytes: 1_000_000 } },
  medium: { steps: 6000, config: { d_model: 128, n_layer: 4, n_head: 4, vocab_size: 1024, lr: 0.002, max_bytes: 4_000_000 } },
};
const MAX_TEXT_BYTES = 8_000_000;
const AUTOSAVE_MS = 30_000;
const PHASE_NAMES = { pretrain: "pre-training", finetune: "fine-tuning", dpo: "preference tuning", loaded: "opened model" };

const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "–");
const fmtInt = (v) => Math.round(v).toLocaleString();
const pct = (v) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "–");

const newView = (total = 0) => ({ points: [], total });

/** A model fine-tuned on the user's own examples has the bare template "{}" (the same marker the command-line tool writes). */
const isCustom = () => state.template === "{}";

/** Read a .jsonl text the way the engine will: one {"prompt","response"} per line. Returns { examples, error }. */
function parseExamples(text) {
  const examples = [];
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    let v;
    try { v = JSON.parse(lines[i]); } catch (e) { return { examples, error: `line ${i + 1} is not valid JSON (${e.message})` }; }
    if (typeof v?.prompt !== "string") return { examples, error: `line ${i + 1} has no "prompt" text` };
    if (typeof v?.response !== "string") return { examples, error: `line ${i + 1} has no "response" text` };
    examples.push(v);
  }
  return { examples, error: examples.length ? null : "the file has no examples" };
}

const MIN_EXAMPLES = 20;

/** The lowest held-out loss seen so far in a view, with the step it happened at (null if none yet). */
function minVal(v) {
  let best = null;
  for (const p of v.points) if (Number.isFinite(p.val) && (!best || p.val < best.val)) best = { step: p.step, val: p.val };
  return best;
}

/** Words of caution for a custom fine-tune whose held-out loss has risen since its best point (a sign of memorising). */
function memorisingNote() {
  const v = state.views.finetune;
  const best = minVal(v);
  const last = v.points[v.points.length - 1];
  if (!best || !last || !(last.val > best.val * 1.1)) return "";
  return ` Warning: the held-out loss was lowest at step ${fmtInt(best.step)} (${best.val.toFixed(2)}) and has risen to ${last.val.toFixed(2)} since, so the model is memorising your examples rather than learning the pattern. More examples, or fewer steps, would help.`;
}
let custom = { text: "", examples: [], error: null, name: "" };   // the chosen examples file

/** 40 simple opposites: enough to try the feature without writing a file first. */
function sampleJsonl() {
  const pairs = ["hot cold", "up down", "big small", "fast slow", "happy sad", "day night", "old new", "hard soft", "light dark", "open closed",
    "full empty", "loud quiet", "near far", "rich poor", "wet dry", "clean dirty", "early late", "young old", "high low", "strong weak",
    "good bad", "long short", "wide narrow", "thick thin", "heavy light", "sweet sour", "warm cool", "right wrong", "inside outside", "front back",
    "push pull", "win lose", "buy sell", "love hate", "start stop", "come go", "give take", "laugh cry", "true false", "yes no"];
  return pairs.map((p) => { const [a, b] = p.split(" "); return JSON.stringify({ prompt: `Opposite of ${a}:`, response: ` ${b}` }); }).join("\n") + "\n";
}
const state = {
  text: null,
  ready: false,
  training: false,
  phase: "none",       // none | pretrain | finetune | loaded
  step: 0,
  finished: { pretrain: false, finetune: false, dpo: false },
  views: { pretrain: newView(), finetune: newView(), dpo: newView() },
  viewPhase: "pretrain",
  samples: [],         // { phase, step, text }, newest first
  ft: { before: null, after: null },
  dpo: { before: null, after: null, info: null, accuracy: null },
  template: null,      // prompt layout for "Ask", when the model is fine-tuned
  meta: null,
  savedRecord: null,
  pendingDownload: null,
  genId: 0,
  askId: 0,
};
window.__tlb = state; // handy for debugging in the console

/** A summary of the state in `<body data-tlb>`, readable by automated tests (the DOM is shared with test drivers). */
function publishState() {
  const last = (v) => v.points[v.points.length - 1] ?? null;
  document.body.dataset.tlb = JSON.stringify({
    phase: state.phase, step: state.step, ready: state.ready, training: state.training, finished: state.finished,
    pre: state.views.pretrain.points.length, ft: state.views.finetune.points.length, dpoPoints: state.views.dpo.points.length,
    lastPre: last(state.views.pretrain), lastFt: last(state.views.finetune), lastDpo: last(state.views.dpo),
    dpoAccuracy: state.dpo.accuracy, dpoInfo: state.dpo.info ? { pairs: state.dpo.info.pairs, good: state.dpo.info.mined_good, samples: state.dpo.info.mined_samples } : null,
    template: state.template, custom: isCustom(), samples: state.samples.length, workers: pool ? pool.n : 0,
    customFile: { examples: custom.examples.length, error: custom.error }, ftMinVal: minVal(state.views.finetune),
  });
}

// ---------- small UI helpers ----------

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}
const view = () => state.views[state.viewPhase];

function syncUI() {
  publishState();
  const { ready, training, phase } = state;
  const canTrainMore = ready && !training && ["pretrain", "finetune", "dpo"].includes(phase) && !state.finished[phase];
  $("btn-load").disabled = training;
  $("btn-train").disabled = !canTrainMore;
  $("btn-train").textContent = state.step > 0 && canTrainMore ? "Resume" : "Train";
  $("btn-pause").disabled = !training;
  $("btn-ft").disabled = !(ready && !training && phase === "pretrain" && state.step >= 100);
  $("btn-gen").disabled = !ready;
  $("btn-ask").disabled = !(ready && state.template);
  $("btn-dl-model").disabled = !ready;
  $("btn-dl-tok").disabled = !ready;
  $("btn-dl-sft").disabled = !(ready && state.template);
  $("btn-save").disabled = !(ready && phase !== "loaded");
  $("btn-open").disabled = training;
  $("tab-finetune").disabled = !(state.views.finetune.points.length || phase === "finetune" || phase === "dpo");
  $("tab-dpo").disabled = !(state.views.dpo.points.length || phase === "dpo");
  $("btn-dpo").disabled = !(ready && !training && phase === "finetune" && state.finished.finetune && !isCustom());
  document.querySelectorAll('input[name="ft-source"]').forEach((r) => (r.disabled = training || phase !== "pretrain"));
  $("ft-file").disabled = training;

  $("ft-hint").textContent =
    phase === "pretrain" ? (state.step >= 100 ? "Ready: this keeps your trained model and teaches it the task." : "Train for at least 100 steps first.")
    : phase === "finetune" || phase === "dpo" ? "This model has been fine-tuned. Prepare a new model to start over."
    : "Available after you train a model here.";
  $("dpo-hint").textContent =
    isCustom() && phase !== "pretrain" ? "Preference tuning needs an automatic check of which answers are good, and your own examples have none. It is available for the built-in word task."
    : phase === "finetune" ? (state.finished.finetune ? "Ready: this keeps the fine-tuned model and teaches it from its own good and bad answers." : "Available once fine-tuning has finished.")
    : phase === "dpo" ? "This model has been through preference tuning. Prepare a new model to start over."
    : "Available once fine-tuning has finished.";
  $("ask-hint").textContent = isCustom()
    ? "Type a prompt written the way your examples were, and the model answers it."
    : state.template
    ? "Type a word and the model writes a sentence using it."
    : "Fine-tune the model first, then type a word and it writes a sentence using it.";
  $("ask-label").textContent = isCustom() ? "Your prompt" : "A word";
}

// ---------- the worker pool ----------
// One leader (talks to this page) plus helpers (talk to the leader). All hold identical models.

let pool = null;
let gatherState = null;

function workerCount() {
  const cores = navigator.hardwareConcurrency || 4;
  const choice = $("workers").value;
  return choice === "auto" ? clamp(Math.floor(cores / 2), 1, 8) : Number(choice);
}

function fillWorkerChoices() {
  const cores = navigator.hardwareConcurrency || 4;
  const auto = clamp(Math.floor(cores / 2), 1, 8);
  const options = [["auto", `Auto (${auto})`]];
  for (const n of [1, 2, 4, 8]) if (n <= cores) options.push([String(n), n === 1 ? "1 (no helpers)" : String(n)]);
  $("workers").innerHTML = options.map(([v, label]) => `<option value="${v}">${label}</option>`).join("");
}

function destroyPool() {
  if (!pool) return;
  pool.leader.terminate();
  pool.helpers.forEach((h) => h.terminate());
  pool = null;
  gatherState = null;
}

function makePool(n) {
  destroyPool();
  const leader = new Worker("worker.js", { type: "module" });
  leader.onmessage = onLeaderMessage;
  leader.onerror = onWorkerError;
  const helpers = [];
  for (let i = 1; i < n; i++) {
    const helper = new Worker("worker.js", { type: "module" });
    const channel = new MessageChannel(); // a private line between the leader and this helper
    leader.postMessage({ type: "attach", port: channel.port1 }, [channel.port1]);
    helper.postMessage({ type: "helperInit", port: channel.port2 }, [channel.port2]);
    helper.onmessage = onHelperMessage;
    helper.onerror = onWorkerError;
    helpers.push(helper);
  }
  pool = { leader, helpers, n };
}

const broadcast = (msg) => {
  pool.leader.postMessage(msg);
  pool.helpers.forEach((h) => h.postMessage(msg));
};

/** Wait until the leader has answered AND every helper has reported ready. */
function gather(leaderOnly = false) {
  return new Promise((resolve, reject) => {
    gatherState = { leaderMsg: null, left: leaderOnly ? 0 : pool.helpers.length, resolve, reject };
  });
}
function gatherCheck() {
  const g = gatherState;
  if (g && g.leaderMsg && g.left === 0) {
    gatherState = null;
    g.resolve(g.leaderMsg);
  }
}
function gatherFail(message) {
  if (!gatherState) return false;
  const g = gatherState;
  gatherState = null;
  g.reject(new Error(message));
  return true;
}

function onHelperMessage(event) {
  const m = event.data;
  if (m.type === "helperReady" && gatherState) {
    gatherState.left--;
    gatherCheck();
  } else if (m.type === "helperError") {
    if (!gatherFail(m.message)) onError(m.message);
  }
}

function onWorkerError(e) {
  const message = e.message || "a worker could not start (open this page through a web server, not as a file)";
  if (!gatherFail(message)) onError(message);
}

function onError(message) {
  state.training = false;
  setStatus(message, true);
  syncUI();
}

// ---------- loading text and creating a session ----------

for (const radio of document.querySelectorAll('input[name="source"]')) {
  radio.addEventListener("change", () => {
    const useFile = document.querySelector('input[name="source"]:checked').value === "file";
    $("file").hidden = !useFile;
    $("file-name").hidden = !useFile;
  });
}
$("file").addEventListener("change", () => {
  const f = $("file").files[0];
  $("file-name").textContent = f ? `${f.name} (${fmtInt(f.size / 1024)} KB)` : "";
});

async function loadText() {
  if (document.querySelector('input[name="source"]:checked').value === "sample") {
    const res = await fetch("data/tinystories-8mb.txt");
    if (!res.ok) throw new Error("Could not load the sample text (web/data/tinystories-8mb.txt).");
    return res.text();
  }
  const file = $("file").files[0];
  if (!file) throw new Error("Choose a text file first.");
  return file.slice(0, MAX_TEXT_BYTES).text();
}

function applyPreset() {
  const preset = PRESETS[$("preset").value];
  $("steps").value = preset.steps;
  const tokens = fmtInt(preset.config.max_bytes / 1000);
  $("preset-hint").textContent = `Recommended: ${fmtInt(preset.steps)} steps, learning from the first ${tokens} KB of text.`;
}
$("preset").addEventListener("change", applyPreset);

function resetForNewSession(phase, total) {
  state.ready = false;
  state.training = false;
  state.phase = phase;
  state.step = 0;
  state.finished = { pretrain: false, finetune: false, dpo: false };
  state.views = { pretrain: newView(total), finetune: newView(), dpo: newView() };
  state.viewPhase = "pretrain";
  state.samples = [];
  state.ft = { before: null, after: null };
  state.dpo = { before: null, after: null, info: null, accuracy: null };
  state.template = null;
  showTab("pretrain");
  renderSamples();
  renderFtTable();
  renderDpoTable();
}

$("btn-load").addEventListener("click", async () => {
  try {
    const preset = PRESETS[$("preset").value];
    const steps = clamp(Number($("steps").value) || preset.steps, 100, 100000);
    resetForNewSession("pretrain", steps);
    syncUI();
    setStatus("Reading the text…");
    state.text = await loadText();
    makePool(workerCount());
    const ready = gather();
    broadcast({ type: "init", text: state.text, config: { ...preset.config, steps } });
    onSessionReady(await ready);
  } catch (err) {
    setStatus(err.message, true);
  }
});

function onSessionReady(msg) {
  state.meta = msg.meta;
  state.phase = msg.meta.phase;
  state.ready = true;
  state.training = false;
  state.step = msg.meta.step;
  if (state.phase === "pretrain") state.views.pretrain.total = msg.meta.total_steps;
  state.template = msg.meta.sft?.template ?? state.template;
  const workers = pool ? pool.n : 1;
  const where = workers > 1 ? ` Training will use ${workers} workers.` : "";
  const how = msg.setupMs ? `, set up in ${fmt(msg.setupMs / 1000, 1)} s` : "";
  setStatus(
    state.phase === "loaded"
      ? `Opened a model with ${fmtInt(msg.meta.params)} parameters (trained for ${fmtInt(msg.meta.base_steps)} steps). It can generate text${state.template ? " and answer questions" : ""} but not train.`
      : `Ready: ${fmtInt(msg.meta.params)} parameters, ${fmtInt(msg.meta.n_tokens)} tokens${how}.${where} Press Train.`
  );
  renderAll();
  syncUI();
}

// ---------- training ----------

$("btn-train").addEventListener("click", () => {
  state.training = true;
  pool.leader.postMessage({ type: "start", prompt: $("prompt").value.trim() || "Once upon a time", word: $("ask-word").value.trim() || "ball" });
  setStatus("Training…");
  syncUI();
});
$("btn-pause").addEventListener("click", () => pool.leader.postMessage({ type: "pause" }));

$("btn-ft").addEventListener("click", async () => {
  try {
    const steps = clamp(Number($("ft-steps").value) || 1500, 100, 100000);
    const useCustom = ftSource() === "file";
    if (useCustom) {
      if (!custom.text) throw new Error("Choose your examples file first.");
      if (custom.error) throw new Error(`Your examples file: ${custom.error}.`);
      if (custom.examples.length < MIN_EXAMPLES) throw new Error(`Your file has ${custom.examples.length} examples; at least ${MIN_EXAMPLES} are needed (a few are held back for scoring).`);
    }
    state.training = true; // blocks other buttons while the examples are prepared
    syncUI();
    setStatus("Preparing the fine-tuning examples…");
    const ready = gather();
    broadcast({ type: "startFinetune", settings: { steps, lr: 0.001, batch: 16, examples: 20000, seed: 7, custom: useCustom ? custom.text : "" } });
    const msg = await ready; // { meta, before, valLoss }
    state.meta = msg.meta;
    state.phase = "finetune";
    state.step = 0;
    state.template = msg.meta.sft.template;
    // While training, the sample line shows the model answering this prompt: use one of the user's own.
    if (isCustom()) $("ask-word").value = custom.examples[0].prompt;
    state.views.finetune = newView(steps);
    state.ft = { before: msg.before, after: null };
    showTab("finetune");
    renderFtTable();
    const info = msg.meta.sft;
    $("ft-note").hidden = false;
    $("ft-note").textContent =
      `${fmtInt(info.train_examples)} training examples, ${fmtInt(info.val_examples)} held out for scoring.` +
      (info.overlap ? " Note: there was not enough unseen text, so some examples overlap the pre-training text." : "") +
      (isCustom() ? " The held-out examples are a random sample of your file, never trained on." : "");
    pool.leader.postMessage({ type: "start", prompt: $("prompt").value.trim(), word: $("ask-word").value.trim() || "ball" });
    setStatus("Fine-tuning…");
    syncUI();
  } catch (err) {
    state.training = false;
    setStatus(err.message, true);
    syncUI();
  }
});

$("btn-dpo").addEventListener("click", async () => {
  try {
    const steps = clamp(Number($("dpo-steps").value) || 300, 50, 10000);
    const prompts = clamp(Number($("dpo-prompts").value) || 1500, 100, 5000);
    state.training = true; // blocks other buttons while pairs are mined
    syncUI();
    setStatus("Preparing preference tuning…");
    const ready = gather(true); // only the leader answers; it coordinates the helpers itself
    pool.leader.postMessage({ type: "startDpo", settings: { steps, prompts, k: 4, lr: 0.0001, beta: 0.3, sft_weight: 4, batch: 16, mine_temperature: 0.7, seed: 11 } });
    const msg = await ready; // { meta, before, valLoss, accuracy }
    state.meta = msg.meta;
    state.phase = "dpo";
    state.step = 0;
    state.views.dpo = newView(steps);
    state.dpo = { before: msg.before, after: null, info: msg.meta.dpo, accuracy: msg.accuracy };
    showTab("dpo");
    renderDpoTable();
    pool.leader.postMessage({ type: "start", prompt: $("prompt").value.trim(), word: $("ask-word").value.trim() || "ball" });
    setStatus("Preference tuning…");
    syncUI();
  } catch (err) {
    state.training = false;
    setStatus(err.message, true);
    syncUI();
  }
});

// ---------- messages from the leader ----------

function onLeaderMessage(event) {
  const m = event.data;
  switch (m.type) {
    case "status":
      return setStatus(m.message);
    case "ready":
    case "finetuneReady":
    case "dpoReady":
      if (gatherState) {
        gatherState.leaderMsg = m;
        gatherCheck();
      }
      return;
    case "progress":
      return onProgress(m);
    case "sample":
      state.samples.unshift({ phase: m.phase, step: m.step, text: m.text });
      publishState();
      return renderSamples();
    case "paused":
      state.training = false;
      state.step = m.step; // progress updates are throttled, so take the exact step from here
      $("s-step").textContent = `${fmtInt(m.step)} / ${fmtInt(state.views[m.phase]?.total ?? 0)}`;
      setStatus(`Paused at step ${fmtInt(m.step)}.`);
      syncUI();
      return requestSave();
    case "done":
      return onDone(m);
    case "generated":
      $("output").textContent = m.text;
      return ($("btn-gen").disabled = !state.ready);
    case "answered":
      return onAnswered(m);
    case "snapshot":
      return onSnapshot(m);
    case "exported":
      return onExported(m);
    case "error":
      if (!gatherFail(m.message)) onError(m.message);
      return;
  }
}

let drawQueued = false;
function onProgress(m) {
  state.step = m.step;
  const v = state.views[m.phase];
  v.total = m.total;
  v.speed = m.stepsPerSec; // remembered so switching tabs still shows it
  if (m.val !== undefined) v.points.push({ step: m.step, train: m.train, val: m.val });
  if (m.valAcc !== undefined) {
    state.dpo.accuracy = m.valAcc;
    renderDpoTable();
  }
  publishState();
  if (m.phase === state.viewPhase) {
    $("s-step").textContent = `${fmtInt(m.step)} / ${fmtInt(m.total)}`;
    $("s-train").textContent = fmt(m.train);
    if (m.val !== undefined) $("s-val").textContent = fmt(m.val);
    $("s-speed").textContent = `${fmt(m.stepsPerSec, 0)} steps/s`;
    if (m.val !== undefined && !drawQueued) {
      drawQueued = true;
      requestAnimationFrame(() => {
        drawQueued = false;
        renderChart();
      });
    }
  }
  if (state.training && m.stepsPerSec > 0) {
    const left = (m.total - m.step) / m.stepsPerSec;
    const who = m.workers > 1 ? ` (${m.workers} workers)` : "";
    setStatus(`${{ finetune: "Fine-tuning", dpo: "Preference tuning" }[state.phase] ?? "Training"}${who}… about ${left < 90 ? `${Math.ceil(left)} s` : `${Math.ceil(left / 60)} min`} left.`);
  }
}

function onDone(m) {
  if (m.breakdown) console.info("worker time per step (ms):", m.breakdown); // for performance debugging
  document.body.dataset.breakdown = JSON.stringify(m.breakdown ?? null);
  state.training = false;
  state.finished[m.phase] = true;
  state.step = m.step;
  const sync = m.workers > 1 ? (m.syncOk ? ` All ${m.workers} workers finished with identical models.` : " Warning: the workers' models differ!") : "";
  if (m.phase === "dpo" && m.after) {
    state.dpo.after = m.after;
    renderDpoTable();
    const was = state.dpo.before?.uses_word;
    setStatus(`Preference tuning finished (${fmtInt(m.step)} steps).${sync} It now uses the requested word in ${pct(m.after.uses_word)} of answers` + (was !== undefined ? ` (was ${pct(was)}).` : "."), !m.syncOk);
    syncUI();
    return requestSave();
  }
  if (m.phase === "finetune" && m.after) {
    state.ft.after = m.after;
    renderFtTable();
    const result = isCustom()
      ? `It now matches the answer you gave exactly on ${pct(m.after.exact)} of held-out prompts (was ${pct(state.ft.before?.exact)}). Exact matching is strict: compare the examples below.${memorisingNote()}`
      : `It now uses the requested word in ${pct(m.after.uses_word)} of answers.`;
    setStatus(`Fine-tuning finished (${fmtInt(m.step)} steps).${sync} ${result} Try it below.`, !m.syncOk);
  } else {
    setStatus(`Finished ${fmtInt(m.step)} steps.${sync} Try it below, or teach it a task.`, !m.syncOk);
  }
  syncUI();
  requestSave();
}

// ---------- generating text ----------

$("btn-gen").addEventListener("click", () => {
  const prompt = $("prompt").value;
  if (!prompt.trim()) return setStatus("Type the start of a story first.", true);
  $("btn-gen").disabled = true;
  pool.leader.postMessage({ type: "generate", id: ++state.genId, prompt, n: 100, temperature: Number($("temp").value), seed: Math.floor(Math.random() * 2 ** 31) });
});
$("temp").addEventListener("input", () => ($("temp-out").textContent = $("temp").value));

$("btn-ask").addEventListener("click", () => {
  const input = $("ask-word").value.trim();
  if (!input) return setStatus("Type a word first.", true);
  $("btn-ask").disabled = true;
  pool.leader.postMessage({ type: "answer", id: ++state.askId, input, temperature: 0.7, seed: Math.floor(Math.random() * 2 ** 31) });
});

function onAnswered(m) {
  $("answers").replaceChildren(
    ...m.texts.map((t) => {
      const li = document.createElement("li");
      li.textContent = t; // model output is always inserted as text, never as HTML
      return li;
    })
  );
  $("btn-ask").disabled = !(state.ready && state.template);
}

// ---------- saving and resuming ----------

function requestSave() {
  if (pool && state.ready && state.phase !== "loaded") pool.leader.postMessage({ type: "saveState" });
}
$("btn-save").addEventListener("click", requestSave);
setInterval(() => state.training && requestSave(), AUTOSAVE_MS);

async function onSnapshot(m) {
  if (m.unavailable) return;
  try {
    await saveSession({
      savedAt: Date.now(),
      meta: m.meta, model: m.model, optimizer: m.optimizer, tokenizer: m.tokenizer, pairs: m.pairs,
      text: state.text,
      ui: { views: state.views, viewPhase: state.viewPhase, samples: state.samples, ft: state.ft, dpo: state.dpo, finished: state.finished, steps: $("steps").value, preset: $("preset").value },
    });
    const meta = JSON.parse(m.meta);
    $("save-status").textContent = `Saved ${new Date().toLocaleTimeString()} (${PHASE_NAMES[meta.phase]}, step ${fmtInt(meta.step)}).`;
  } catch (err) {
    $("save-status").textContent = `Could not save in this browser: ${err.message}`;
  }
}

async function checkForSavedSession() {
  try {
    const record = await loadSession();
    if (!record) return;
    state.savedRecord = record;
    const meta = JSON.parse(record.meta);
    $("resume-text").textContent =
      `You have a saved session from ${new Date(record.savedAt).toLocaleString()}: ${PHASE_NAMES[meta.phase]}, ` +
      `step ${fmtInt(meta.step)} of ${fmtInt(meta.total_steps)}, ${fmtInt(meta.params)} parameters.`;
    $("resume-banner").hidden = false;
  } catch {
    $("save-status").textContent = "Automatic saving is not available in this browser.";
  }
}

$("btn-resume").addEventListener("click", async () => {
  const rec = state.savedRecord;
  if (!rec) return;
  try {
    $("resume-banner").hidden = true;
    setStatus("Restoring your saved session…");
    state.text = rec.text;
    makePool(workerCount());
    const ready = gather();
    broadcast({ type: "restore", text: rec.text, meta: rec.meta, model: rec.model, optimizer: rec.optimizer, tokenizer: rec.tokenizer, pairs: rec.pairs });
    const msg = await ready;
    const ui = rec.ui ?? {};
    // Older saves have no preference-tuning entries, so fill in defaults.
    state.views = { pretrain: newView(), finetune: newView(), dpo: newView(), ...ui.views };
    state.samples = ui.samples ?? [];
    state.ft = ui.ft ?? { before: null, after: null };
    state.dpo = ui.dpo ?? { before: null, after: null, info: null, accuracy: null };
    state.finished = { pretrain: false, finetune: false, dpo: false, ...ui.finished };
    if (ui.steps) $("steps").value = ui.steps;
    if (ui.preset) $("preset").value = ui.preset;
    state.template = null;
    state.phase = msg.meta.phase;
    state.viewPhase = ["finetune", "dpo"].includes(msg.meta.phase) ? msg.meta.phase : "pretrain";
    onSessionReady(msg);
    showTab(state.viewPhase);
    renderSamples();
    renderFtTable();
    renderDpoTable();
    setStatus(`Resumed at step ${fmtInt(msg.meta.step)} of ${fmtInt(msg.meta.total_steps)}. Press Train to continue.`);
  } catch (err) {
    $("resume-banner").hidden = false;
    setStatus(`Could not resume: ${err.message}`, true);
  }
});

$("btn-discard").addEventListener("click", async () => {
  try {
    await deleteSession();
  } catch { /* nothing to delete */ }
  state.savedRecord = null;
  $("resume-banner").hidden = true;
});

// ---------- downloads and opening files ----------

function saveFile(name, data, type) {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function requestDownload(which) {
  state.pendingDownload = which;
  pool.leader.postMessage({ type: "export" });
}
$("btn-dl-model").addEventListener("click", () => requestDownload("model"));
$("btn-dl-tok").addEventListener("click", () => requestDownload("tokenizer"));
$("btn-dl-sft").addEventListener("click", () => requestDownload("sft"));

function onExported(m) {
  const which = state.pendingDownload;
  state.pendingDownload = null;
  if (which === "model") return saveFile("model.safetensors", m.model, "application/octet-stream");
  if (which === "tokenizer") return saveFile("tokenizer.json", m.tokenizer, "application/json");
  // sft.json has the same fields the command-line tool writes, so `tlm chat` can use it.
  const info = m.meta.sft;
  saveFile("sft.json", JSON.stringify({
    base: "pre-trained in the browser",
    task: isCustom() ? "jsonl" : "use-the-word",
    template: state.template,
    stop: "<|endoftext|>",
    steps: info?.settings.steps ?? 0,
    train_examples: info?.train_examples ?? 0,
    word_rate_before: isCustom() ? null : state.ft.before?.uses_word ?? null,
    word_rate_after: isCustom() ? null : state.ft.after?.uses_word ?? null,
  }, null, 2), "application/json");
}

$("btn-open").addEventListener("click", async () => {
  try {
    const files = [...$("open-files").files];
    const model = files.find((f) => f.name.endsWith(".safetensors"));
    const tokenizer = files.find((f) => f.name === "tokenizer.json");
    const sftFile = files.find((f) => f.name === "sft.json");
    if (!model || !tokenizer) throw new Error("Choose both model.safetensors and tokenizer.json.");
    let template = null;
    if (sftFile) {
      template = JSON.parse(await sftFile.text()).template;
      if (typeof template !== "string" || !template.includes("{}")) throw new Error("sft.json has no usable \"template\".");
    }
    resetForNewSession("loaded", 0);
    syncUI();
    setStatus("Opening the model…");
    makePool(1);
    const ready = gather();
    pool.leader.postMessage({ type: "loadModel", model: new Uint8Array(await model.arrayBuffer()), tokenizer: await tokenizer.text(), template });
    state.template = template;
    onSessionReady(await ready);
  } catch (err) {
    setStatus(err.message, true);
  }
});

// ---------- choosing your own examples ----------

const ftSource = () => document.querySelector('input[name="ft-source"]:checked').value;

function showFileNote() {
  const note = $("ft-file-note");
  note.classList.remove("error");
  if (!custom.text) {
    note.textContent = "Choose a file with one example per line.";
  } else if (custom.error) {
    note.textContent = `${custom.name}: ${custom.error}.`;
    note.classList.add("error");
  } else if (custom.examples.length < MIN_EXAMPLES) {
    note.textContent = `${custom.name}: ${custom.examples.length} examples. At least ${MIN_EXAMPLES} are needed.`;
    note.classList.add("error");
  } else {
    const long = custom.examples.filter((e) => e.prompt.length + e.response.length > 240).length;
    note.textContent = `${custom.name}: ${fmtInt(custom.examples.length)} examples found.` +
      (long ? ` ${long} look long for the 64-token window and may be skipped.` : "");
  }
  publishState();
}

document.querySelectorAll('input[name="ft-source"]').forEach((r) => r.addEventListener("change", () => {
  $("ft-custom").hidden = ftSource() !== "file";
  showFileNote();
}));

$("ft-file").addEventListener("change", async () => {
  const f = $("ft-file").files[0];
  if (!f) {
    custom = { text: "", examples: [], error: null, name: "" };
    return showFileNote();
  }
  if (f.size > 1_000_000) {
    custom = { text: "", examples: [], error: "the file is larger than 1 MB", name: f.name };
    return showFileNote();
  }
  const text = await f.text();
  const parsed = parseExamples(text);
  custom = { text, examples: parsed.examples, error: parsed.error, name: f.name };
  showFileNote();
});

$("btn-sample-jsonl").addEventListener("click", () => saveFile("opposites-sample.jsonl", sampleJsonl(), "application/x-ndjson"));

/** Held-out examples side by side: what you wrote and what the model says. Only for the user's own examples. */
function renderCompare() {
  const e = state.ft.after ?? state.ft.before;
  const show = isCustom() && !!e?.samples?.length;
  $("ft-compare").hidden = !show;
  if (!show) return;
  $("ft-compare-title").textContent = state.ft.after ? "Held-out examples after fine-tuning: your answer and the model's" : "Held-out examples before fine-tuning: your answer and the model's";
  $("ft-compare-list").replaceChildren(...e.samples.map((x) => {
    const li = document.createElement("li");
    const row = (k, v) => {
      const label = document.createElement("span");
      label.className = "k";
      label.textContent = k;
      li.append(label, document.createTextNode(v)); // text only, never HTML
    };
    row("Prompt", x.prompt);
    row("Your answer", x.reference.trim());
    row("Model's answer", x.answer.trim() || "(nothing)");
    return li;
  }));
}

// ---------- fine-tuning results table ----------

function renderFtTable() {
  const { before, after } = state.ft;
  $("ft-table").hidden = !before;
  $("ft-note").hidden = !before;
  const own = isCustom();
  $("ft-row-word").textContent = own ? "Matches your answer exactly" : "Uses the requested word";
  $("ft-row-stops").textContent = own ? "Finishes its answer" : "Finishes its sentence";
  $("ft-row-chance").hidden = own;
  const cells = (e, which) => {
    $(`ft-${which}-word`).textContent = e ? pct(own ? e.exact : e.uses_word) : "–";
    $(`ft-${which}-chance`).textContent = e ? pct(e.chance) : "–";
    $(`ft-${which}-stops`).textContent = e ? pct(e.stops) : "–";
  };
  renderCompare();
  cells(before, "before");
  cells(after, "after");
  $("ft-after-word").classList.toggle("good", !!after);
}

// ---------- preference-tuning results table ----------

function renderDpoTable() {
  const { before, after, info, accuracy } = state.dpo;
  $("dpo-table").hidden = !before;
  $("dpo-note").hidden = !info;
  const cells = (e, which) => {
    $(`dpo-${which}-word`).textContent = e ? pct(e.uses_word) : "–";
    $(`dpo-${which}-chance`).textContent = e ? pct(e.chance) : "–";
    $(`dpo-${which}-stops`).textContent = e ? pct(e.stops) : "–";
  };
  cells(before, "before");
  cells(after, "after");
  $("dpo-after-word").classList.toggle("good", !!after);
  if (info) {
    const goodPct = info.mined_samples ? (100 * info.mined_good) / info.mined_samples : 0;
    $("dpo-note").textContent =
      `${fmtInt(info.mined_samples)} answers sampled; ${goodPct.toFixed(0)}% passed the check. ` +
      `${fmtInt(info.pairs)} pairs (${fmtInt(info.val_pairs)} held out).` +
      // Before any training the model is the reference, so the accuracy is 0 by construction: say nothing yet.
      (Number.isFinite(accuracy) && (state.step > 0 || after) ? ` On held-out pairs the model now prefers the good answer ${(accuracy * 100).toFixed(0)}% of the time.` : "") +
      (info.pairs < 150 ? " Only a few pairs were found, so expect noisy results: sample more prompts, or fine-tune longer so answers are more mixed." : "");
  }
}

// ---------- samples ----------

function renderSamples() {
  const list = $("samples");
  if (!state.samples.length) {
    list.innerHTML = '<li class="empty">Samples appear here as training progresses.</li>';
    return;
  }
  list.replaceChildren(
    ...state.samples.map((s) => {
      const li = document.createElement("li");
      const at = document.createElement("span");
      at.className = "at";
      at.textContent = `${{ finetune: "Fine-tuning", dpo: "Preference tuning" }[s.phase] ?? "Pre-training"}, step ${fmtInt(s.step)}`;
      li.append(at, document.createTextNode(s.text)); // text nodes: model output is never parsed as HTML
      return li;
    })
  );
}

// ---------- phase tabs ----------

const TAB_TEXT = {
  pretrain: {
    train: "Train loss", val: "Validation loss",
    units: "Loss is in bits per byte of text: lower is better, 8 means no idea at all.",
  },
  finetune: {
    train: "Train loss (answers)", val: "Validation loss (answers)",
    units: "Loss is counted only on the answer, in nats per token: lower is better. The instruction itself is not scored.",
  },
  dpo: {
    train: "Train loss (pairs)", val: "Validation loss (pairs)",
    units: "Preference loss on pairs of answers: it starts at 0.693 (no preference yet) and falls as the model learns to favour the good answer over the bad one.",
  },
};

function showTab(phase) {
  state.viewPhase = phase;
  for (const p of ["pretrain", "finetune", "dpo"]) $(`tab-${p}`).setAttribute("aria-selected", String(p === phase));
  $("s-train-label").textContent = TAB_TEXT[phase].train;
  $("s-val-label").textContent = TAB_TEXT[phase].val;
  $("units").textContent = TAB_TEXT[phase].units;
  renderAll();
}
for (const p of ["pretrain", "finetune", "dpo"]) {
  $(`tab-${p}`).addEventListener("click", () => !$(`tab-${p}`).disabled && showTab(p));
}
function renderAll() {
  const v = view();
  const last = v.points[v.points.length - 1];
  const stepNow = state.viewPhase === state.phase ? state.step : v.points.length ? last.step : 0;
  $("s-step").textContent = `${fmtInt(stepNow)}${v.total ? ` / ${fmtInt(v.total)}` : ""}`;
  $("s-train").textContent = last ? fmt(last.train) : "–";
  $("s-val").textContent = last ? fmt(last.val) : "–";
  $("s-speed").textContent = v.speed ? `${fmt(v.speed, 0)} steps/s` : "–";
  renderChart();
}

// ---------- the chart ----------

// The chart is drawn at its real on-screen pixel width, so text stays 11-12px on a phone.
let W = 720, H = 310;
// The top margin leaves room for the y-axis title ABOVE the plot, so no curve can run through it.
const M = { left: 46, right: 92, top: 30, bottom: 40 };

function fitToContainer() {
  const width = Math.round($("chart").getBoundingClientRect().width);
  if (!width) return false; // hidden (table view is showing); keep the last size
  W = Math.max(300, width);
  const narrow = W < 520;
  H = narrow ? 260 : 310;
  M.left = narrow ? 38 : 46;
  M.right = narrow ? 84 : 92;
  $("chart").setAttribute("viewBox", `0 0 ${W} ${H}`);
  return true;
}
new ResizeObserver(() => requestAnimationFrame(renderChart)).observe($("chart-wrap"));

function niceTicks(lo, hi, n) {
  const span = hi - lo || 1;
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const ticks = [];
  // Keep adding ticks until one reaches or passes `hi`, so the axis always covers the data.
  for (let v = Math.floor(lo / step) * step; ; v += step) {
    ticks.push(+v.toFixed(10));
    if (v >= hi - step * 0.001) break;
  }
  return { ticks, step };
}

let geometry = null; // remembered for the hover handler

function renderChart() {
  const svg = $("chart");
  if (!fitToContainer()) return;
  const v = view();
  const pts = v.points;
  if (pts.length < 2) {
    geometry = null;
    const msg = state.ready ? "Press Train to see the loss curve" : "Prepare a model, then train it to see the loss curve";
    svg.innerHTML = `<text class="empty" x="${W / 2}" y="${H / 2}" text-anchor="middle">${msg}</text>`;
    renderTable();
    return;
  }

  const all = pts.flatMap((p) => [p.train, p.val]);
  // Pad the range a little so lines never sit on the axis.
  const lo = Math.min(...all), hi = Math.max(...all);
  // (Every quantity charted here is a loss, which cannot be negative, so never pad below zero.)
  const yt = niceTicks(Math.max(0, lo - 0.06 * (hi - lo)), hi + 0.02 * (hi - lo), 4);
  const y0 = yt.ticks[0], y1 = yt.ticks[yt.ticks.length - 1];
  const total = v.total || pts[pts.length - 1].step;
  const xt = niceTicks(0, total, 4);
  const px = (s) => M.left + (s / total) * (W - M.left - M.right);
  const py = (val) => H - M.bottom - ((val - y0) / (y1 - y0 || 1)) * (H - M.top - M.bottom);
  const yDec = Math.max(0, -Math.floor(Math.log10(yt.step)));

  let out = "";
  for (const t of yt.ticks) {
    out += `<line class="grid" x1="${M.left}" x2="${W - M.right}" y1="${py(t)}" y2="${py(t)}"/>` +
           `<text x="${M.left - 8}" y="${py(t) + 4}" text-anchor="end">${t.toFixed(yDec)}</text>`;
  }
  out += `<line class="axis" x1="${M.left}" x2="${W - M.right}" y1="${H - M.bottom}" y2="${H - M.bottom}"/>`;
  for (const t of xt.ticks) {
    if (t > total) continue;
    out += `<text x="${px(t)}" y="${H - M.bottom + 16}" text-anchor="middle">${fmtInt(t)}</text>`;
  }
  const yTitle = { pretrain: "bits per byte (lower is better)", finetune: "loss on answers (lower is better)", dpo: "preference loss (lower is better)" }[state.viewPhase];
  out += `<text x="${M.left}" y="${M.top - 14}">${yTitle}</text>`;
  out += `<text x="${(M.left + W - M.right) / 2}" y="${H - 6}" text-anchor="middle">training step</text>`;

  const path = (key) => pts.map((p, i) => `${i ? "L" : "M"}${px(p.step).toFixed(1)},${py(p[key]).toFixed(1)}`).join("");
  out += `<path class="line s1" d="${path("train")}"/><path class="line s2" d="${path("val")}"/>`;

  // End-of-line dots with direct labels; nudge apart if they would overlap.
  const last = pts[pts.length - 1];
  let yTrain = py(last.train), yVal = py(last.val);
  if (Math.abs(yTrain - yVal) < 14) {
    const mid = (yTrain + yVal) / 2;
    [yTrain, yVal] = yTrain <= yVal ? [mid - 7, mid + 7] : [mid + 7, mid - 7];
  }
  out += `<circle class="dot s1" r="4" cx="${px(last.step)}" cy="${py(last.train)}"/>` +
         `<circle class="dot s2" r="4" cx="${px(last.step)}" cy="${py(last.val)}"/>` +
         `<text class="label" x="${px(last.step) + 10}" y="${yTrain + 4}">Train</text>` +
         `<text class="label" x="${px(last.step) + 10}" y="${yVal + 4}">Validation</text>`;

  // Hover layer: crosshair + two marker dots, moved by the pointer handler.
  out += `<g id="hover" hidden>` +
         `<line class="cross" id="h-line" y1="${M.top}" y2="${H - M.bottom}"/>` +
         `<circle class="dot s1" id="h-d1" r="5"/><circle class="dot s2" id="h-d2" r="5"/></g>` +
         `<rect id="h-hit" x="${M.left}" y="${M.top}" width="${W - M.left - M.right}" height="${H - M.top - M.bottom}" fill="transparent"/>`;

  svg.innerHTML = out;
  geometry = { px, py };
  renderTable();
}

// Hover: find the evaluation nearest the pointer and show both values.
$("chart").addEventListener("pointermove", (e) => {
  const pts = view().points;
  if (!geometry || pts.length < 2) return;
  const svg = $("chart");
  const rect = svg.getBoundingClientRect();
  const vbX = ((e.clientX - rect.left) / rect.width) * W;
  let best = pts[0], bestD = Infinity;
  for (const p of pts) {
    const d = Math.abs(geometry.px(p.step) - vbX);
    if (d < bestD) { best = p; bestD = d; }
  }
  const x = geometry.px(best.step);
  svg.querySelector("#hover").removeAttribute("hidden");
  svg.querySelector("#h-line").setAttribute("x1", x);
  svg.querySelector("#h-line").setAttribute("x2", x);
  for (const [id, key] of [["#h-d1", "train"], ["#h-d2", "val"]]) {
    svg.querySelector(id).setAttribute("cx", x);
    svg.querySelector(id).setAttribute("cy", geometry.py(best[key]));
  }
  const tip = $("tooltip");
  tip.hidden = false;
  tip.innerHTML =
    `<div class="t-step">Step ${fmtInt(best.step)}</div>` +
    `<div class="t-row"><span class="swatch s1"></span>Train <b>${fmt(best.train, 3)}</b></div>` +
    `<div class="t-row"><span class="swatch s2"></span>Validation <b>${fmt(best.val, 3)}</b></div>`;
  const scale = rect.width / W;
  const left = x * scale + 12;
  tip.style.left = `${left + tip.offsetWidth > rect.width ? x * scale - tip.offsetWidth - 12 : left}px`;
  tip.style.top = `${Math.max(0, (M.top + 4) * scale)}px`;
});
$("chart").addEventListener("pointerleave", () => {
  $("tooltip").hidden = true;
  $("chart").querySelector("#hover")?.setAttribute("hidden", "");
});

// ---------- table view (same data, for anyone who prefers numbers) ----------

$("btn-table").addEventListener("click", () => {
  const showTable = $("table-wrap").hidden;
  $("table-wrap").hidden = !showTable;
  $("chart-wrap").hidden = showTable;
  $("btn-table").setAttribute("aria-pressed", String(showTable));
  $("btn-table").textContent = showTable ? "Show as chart" : "Show as table";
  renderTable();
});

function renderTable() {
  if ($("table-wrap").hidden) return;
  const rows = view().points.map((p) => `<tr><td>${fmtInt(p.step)}</td><td>${fmt(p.train, 3)}</td><td>${fmt(p.val, 3)}</td></tr>`);
  $("table").querySelector("tbody").innerHTML = rows.join("") || '<tr><td colspan="3">No data yet.</td></tr>';
}

// ---------- start-up ----------

fillWorkerChoices();
applyPreset();
syncUI();
renderChart();
checkForSavedSession();
