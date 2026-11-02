// Chapter 5 widgets: apply temperature and top-k to the real model's next-token scores, roll the die,
// and compare whole generations at three temperatures.
import init, { Session, next_logits, vocab_bytes } from "../pkg/tlm_wasm.js";
import { $, chip, fetchSample, nextPaint } from "./common.js";

const CONFIG = { vocab_size: 512, d_model: 32, n_layer: 2, n_head: 2, ctx: 64, batch: 16, steps: 100000, max_bytes: 1000000 };
const SHOWN = 12;                      // bars drawn
const TOPK_ALL = 41;                   // the slider's last position means "no limit"
const GEN_TEMPS = [0.3, 0.8, 1.6];
const GEN_TOKENS = 60;

let session = null;
let sampleText = "";
let trained = false;
let vocab = [];                        // token id -> bytes
let next = null;                       // { tokens, truncated, logits } for the prompt
let dist = null;                       // { probs, order } after temperature and top-k
let counts = null;                     // Float of rolls per token id after "Roll 1,000 times"
let lastRoll = null;
let seed = 1;
let gens = ["", "", ""];

const setStatus = (msg, error = false) => { $("status").textContent = msg; $("status").classList.toggle("error", error); };
const temperature = () => Number($("temp").value);
const topK = () => (Number($("topk").value) >= TOPK_ALL ? Infinity : Number($("topk").value));

/** Tiny seeded random numbers (mulberry32), so a roll can be repeated. */
function rng(seedValue) {
  let a = seedValue >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---- the maths: scores -> chances ---------------------------------------------

/** Divide the scores by the temperature, keep the best k, softmax. Returns probabilities and the ids best-first. */
function applyDials(logits, T, k) {
  const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a] || a - b);
  const keep = Math.min(order.length, k);
  const top = logits[order[0]] / T;
  const probs = new Array(logits.length).fill(0);
  let total = 0;
  for (let r = 0; r < keep; r++) {
    const id = order[r];
    const e = Math.exp(logits[id] / T - top);          // measured from the top score, so e^x cannot overflow
    probs[id] = e;
    total += e;
  }
  for (let r = 0; r < keep; r++) probs[order[r]] /= total;
  return { probs, order };
}

function entropyBits(probs) {
  let h = 0;
  for (const p of probs) if (p > 0) h -= p * Math.log2(p);
  return h;
}

function draw(probs, rand) {
  let u = rand();
  for (let i = 0; i < probs.length; i++) {
    u -= probs[i];
    if (u < 0 && probs[i] > 0) return i;
  }
  return probs.findIndex((p) => p > 0);                // rounding left a sliver over: take the first possible token
}

// ---- drawing ------------------------------------------------------------------

function refresh() {
  try {
    next = JSON.parse(next_logits(session, $("text").value));
  } catch (e) {
    next = null;
    dist = null;
    $("dist").replaceChildren();
    $("dist-line").textContent = String(e.message ?? e);
    $("stats").textContent = "";
    $("trunc-note").hidden = true;
    publish();
    return;
  }
  $("trunc-note").hidden = !next.truncated;
  $("trunc-note").textContent = next.truncated ? `Your prompt is longer than the model's ${next.tokens.length}-token window, so only its last ${next.tokens.length} tokens are used.` : "";
  counts = null;
  lastRoll = null;
  $("roll-line").textContent = "";
  renderDist();
}

function renderDist() {
  if (!next) return;
  const T = temperature(), k = topK();
  $("temp-out").textContent = T.toFixed(2);
  $("topk-out").textContent = k === Infinity ? "all" : String(k);
  dist = applyDials(next.logits, T, k);
  const { probs, order } = dist;
  const list = $("dist");
  list.replaceChildren();
  const max = probs[order[0]];
  order.slice(0, SHOWN).forEach((id) => {
    const li = document.createElement("li");
    li.append(chip(vocab[id], "c2"));
    const bar = document.createElement("span");
    bar.className = "sim";
    const fill = document.createElement("span");
    fill.style.width = `${(probs[id] / Math.max(max, 1e-9)) * 100 * 0.97}%`;
    bar.append(fill);
    if (counts) {
      const tick = document.createElement("i");
      tick.className = "tick-obs";
      tick.style.left = `${(counts[id] / 1000 / Math.max(max, 1e-9)) * 100 * 0.97}%`;
      tick.title = `came up ${counts[id]} times in 1,000 rolls`;
      bar.append(tick);
    }
    const num = document.createElement("span");
    num.className = "num";
    num.textContent = probs[id] >= 0.0005 ? `${(probs[id] * 100).toFixed(1)}%` : "~0";
    num.title = `raw score (logit) ${next.logits[id].toFixed(2)}`;
    li.append(bar, num);
    list.append(li);
  });
  const shown = order.slice(0, SHOWN).reduce((a, id) => a + probs[id], 0);
  const h = entropyBits(probs);
  const tokens = next.tokens;
  const tail = tokens.slice(-6).map((t) => t.bytes);
  $("dist-line").replaceChildren("After the prompt ending ", ...tail.map((b) => chip(b, "c0")), ", the chances for the next token:");
  let cover = 0, need = 0;
  for (const id of order) {
    cover += probs[id];
    need++;
    if (cover >= 0.9) break;
  }
  $("stats").textContent = `The 12 bars hold ${(shown * 100).toFixed(0)}% of the chances. The favourite has ${(max * 100).toFixed(0)}%. It takes ${need} token${need === 1 ? "" : "s"} to cover 90%. Uncertainty ${h.toFixed(2)} bits, as if choosing evenly among about ${Math.pow(2, h).toFixed(1)} tokens (a flat spread over all ${next.logits.length} would be ${Math.log2(next.logits.length).toFixed(0)} bits).`;
  publish();
}

function rollOnce() {
  if (!dist) return;
  const id = draw(dist.probs, rng(Date.now() + Math.floor(Math.random() * 1e6)));
  lastRoll = id;
  const line = $("roll-line");
  line.replaceChildren("The die picked ", chip(vocab[id], "c1"), ` (it had a ${(dist.probs[id] * 100).toFixed(1)}% chance).`);
  publish();
}

function rollMany() {
  if (!dist) return;
  const rand = rng(12345 + Math.floor(Math.random() * 1e6));
  counts = new Array(dist.probs.length).fill(0);
  for (let i = 0; i < 1000; i++) counts[draw(dist.probs, rand)]++;
  const top = dist.order[0];
  $("roll-line").textContent = `In 1,000 rolls the favourite came up ${counts[top]} times (its chance says about ${Math.round(dist.probs[top] * 1000)}).`;
  renderDist();
}

// ---- whole generations --------------------------------------------------------

function writeAll() {
  seed = (seed * 1103515245 + 12345) % 2147483647;
  const prompt = $("text").value;
  $("seed-out").textContent = `seed ${seed}`;
  GEN_TEMPS.forEach((T, i) => {
    let out;
    try {
      out = session.generate(prompt, GEN_TOKENS, T, seed);
    } catch (e) {
      out = String(e.message ?? e);
    }
    gens[i] = out;
    $("gen-" + i).textContent = out;
  });
  publish();
}

// ---- training -----------------------------------------------------------------

async function train() {
  const total = Number($("steps").value);
  $("btn-train").disabled = true;
  $("steps").disabled = true;
  const bar = $("progress");
  bar.hidden = false;
  bar.value = 0;
  if (trained) {
    setStatus("Starting a fresh model…");
    await nextPaint();
    session.free();
    session = new Session(sampleText, JSON.stringify(CONFIG));
    trained = false;
  }
  const t0 = performance.now();
  let done = 0, loss = 0;
  while (done < total) {
    const sliceStart = performance.now();
    while (done < total && performance.now() - sliceStart < 40) {
      const n = Math.min(2, total - done);
      loss = session.step(n);
      done += n;
    }
    bar.value = done / total;
    setStatus(`Training… step ${done} of ${total} (loss ${loss.toFixed(2)}, ${(done / ((performance.now() - t0) / 1000)).toFixed(0)} steps/s)`);
    await nextPaint();
  }
  bar.hidden = true;
  trained = true;
  $("btn-train").disabled = false;
  $("btn-train").textContent = "Train again";
  $("steps").disabled = false;
  $("btn-gen").disabled = false;
  setStatus(`Done: ${total} steps in ${((performance.now() - t0) / 1000).toFixed(1)} s (loss ${loss.toFixed(2)}). Turn the dials below.`);
  refresh();
  writeAll();
}

function publish() {
  const top = dist ? dist.order.slice(0, SHOWN).map((id) => +dist.probs[id].toFixed(4)) : [];
  document.body.dataset.book = JSON.stringify({
    trained, temp: temperature(), k: topK() === Infinity ? null : topK(),
    top, probSum: dist ? +dist.probs.reduce((a, b) => a + b, 0).toFixed(4) : null,
    nonzero: dist ? dist.probs.filter((p) => p > 0).length : 0,
    favourite: dist ? dist.order[0] : null,
    entropy: dist ? +entropyBits(dist.probs).toFixed(3) : null,
    counts: counts ? counts.reduce((a, b) => a + b, 0) : null,
    favouriteCount: counts && dist ? counts[dist.order[0]] : null,
    lastRoll,
    gens,
    params: session ? session.num_params() : null,
  });
}

// ---- start up -----------------------------------------------------------------

async function main() {
  try {
    await init();
    setStatus("Loading the story sample…");
    const text = await fetchSample();
    if (!text) {
      setStatus("The story sample (web/data/tinystories-8mb.txt) was not found. See the README for how to create it.", true);
      return;
    }
    sampleText = text;
    setStatus("Preparing a tiny model (learning the vocabulary from the stories)…");
    await nextPaint();
    session = new Session(text, JSON.stringify(CONFIG));
    vocab = JSON.parse(vocab_bytes(session));
    $("model-line").textContent = `Model: ${session.num_params().toLocaleString()} numbers; ${CONFIG.n_layer} layers; window of ${CONFIG.ctx} tokens; vocabulary of ${CONFIG.vocab_size}.`;
    $("text").addEventListener("input", () => { refresh(); });
    // Old rolls describe the old chances, so any dial change clears them.
    const dialChanged = () => { counts = null; lastRoll = null; $("roll-line").textContent = ""; renderDist(); };
    $("temp").addEventListener("input", dialChanged);
    $("topk").addEventListener("input", dialChanged);
    $("btn-one").addEventListener("click", rollOnce);
    $("btn-many").addEventListener("click", rollMany);
    $("btn-gen").addEventListener("click", writeAll);
    $("btn-train").addEventListener("click", train);
    $("btn-train").disabled = false;
    refresh();
    setStatus("Ready. The model is still random, so its chances are nearly flat. Press Train, then turn the dials.");
  } catch (e) {
    setStatus("Could not start: " + (e.message ?? e) + ". Build with scripts/build-web.ps1 and serve the web folder.", true);
  }
}
main();
