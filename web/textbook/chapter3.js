// Chapter 3 widgets: look inside the real attention layers of an untrained and a trained tiny model.
import init, { Session, attention_for_text } from "../pkg/tlm_wasm.js";
import { $, chip, label, fetchSample, nextPaint } from "./common.js";

const SVG = "http://www.w3.org/2000/svg";
const CONFIG = { vocab_size: 512, d_model: 32, n_layer: 2, n_head: 2, ctx: 64, batch: 16, steps: 100000, max_bytes: 1000000 };
const SKIP = 3;                       // rows with fewer visible tokens than this are left out of the head summary

let sampleText = "";
let trainedSession = null;            // the live session, trained in place
let untrained = null;                 // a loaded copy of the model as it was at step 0
let trained = false;
let trace = null;                     // latest attention for the text in the box
let layer = 0, head = 0, query = 0;

const svgEl = (name, attrs = {}) => {
  const el = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
};
const setStatus = (msg, error = false) => { $("status").textContent = msg; $("status").classList.toggle("error", error); };
const which = () => document.querySelector('input[name="which"]:checked').value;
const activeSession = () => (which() === "trained" && trainedSession ? trainedSession : untrained);
const tokText = (t) => label(t.bytes).text;

// ---- computing and reading the trace ------------------------------------------

function refresh() {
  const text = $("text").value;
  try {
    trace = JSON.parse(attention_for_text(activeSession(), text));
  } catch (e) {
    trace = null;
    $("heat").replaceChildren();
    $("score-rows").replaceChildren();
    $("row-title").textContent = String(e.message ?? e);
    $("head-rows").replaceChildren();
    publish();
    return;
  }
  const T = trace.tokens.length;
  fillSelect("layer", trace.n_layer, "Layer ");
  fillSelect("head", trace.n_head, "Head ");
  layer = Math.min(layer, trace.n_layer - 1);
  head = Math.min(head, trace.n_head - 1);
  $("layer").value = String(layer);
  $("head").value = String(head);
  query = Math.min(query, T - 1);
  $("trunc-note").hidden = !trace.truncated;
  $("trunc-note").textContent = trace.truncated ? `Your text is longer than the model's ${T}-token window, so only the first ${T} tokens are shown.` : "";
  renderAll();
}

function fillSelect(id, n, prefix) {
  const el = $(id);
  if (el.options.length === n) return;
  el.replaceChildren(...Array.from({ length: n }, (_, i) => new Option(prefix + (i + 1), String(i))));
}

const w = (l, h, i, j) => trace.weights[l][h][i * trace.tokens.length + j];
const s = (l, h, i, j) => trace.scores[l][h][i * trace.tokens.length + j];

function renderAll() {
  renderHeat();
  renderRow();
  renderHeads();
  publish();
}

// ---- heat map -----------------------------------------------------------------

function renderHeat() {
  const svg = $("heat");
  svg.replaceChildren();
  const T = trace.tokens.length;
  const cell = Math.max(9, Math.min(30, Math.floor(520 / T)));
  const showLabels = cell >= 12;
  const left = showLabels ? 96 : 8;
  const top = showLabels ? 96 : 8;
  const W = left + T * cell + 8, H = top + T * cell + 8;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.style.maxWidth = `${W * 1.4}px`;

  for (let i = 0; i < T; i++) {
    const rowG = svgEl("g", { class: "row" + (i === query ? " sel" : ""), "data-row": i });
    for (let j = 0; j < T; j++) {
      const x = left + j * cell, y = top + i * cell;
      if (j > i) {
        rowG.append(svgEl("rect", { x, y, width: cell - 1, height: cell - 1, class: "masked" }));
        continue;
      }
      const wt = w(layer, head, i, j);
      const r = svgEl("rect", { x, y, width: cell - 1, height: cell - 1, class: "cellw", "fill-opacity": Math.min(1, Math.max(0.04, wt)).toFixed(3) });
      const tip = svgEl("title");
      tip.textContent = `"${tokText(trace.tokens[i])}" looks at "${tokText(trace.tokens[j])}": weight ${wt.toFixed(3)} (score ${s(layer, head, i, j).toFixed(2)})`;
      r.append(tip);
      rowG.append(r);
    }
    if (showLabels) {
      const t = svgEl("text", { x: left - 6, y: top + i * cell + cell * 0.7, "text-anchor": "end", class: "ylab" });
      t.textContent = tokText(trace.tokens[i]);
      rowG.append(t);
    }
    // a transparent hit area over the whole row (including its label) makes the row clickable
    rowG.append(svgEl("rect", { x: 0, y: top + i * cell, width: W, height: cell, class: "hit" }));
    svg.append(rowG);
  }
  if (showLabels) {
    for (let j = 0; j < T; j++) {
      const t = svgEl("text", { x: left + j * cell + cell * 0.45, y: top - 6, class: "xlab", transform: `rotate(-55 ${left + j * cell + cell * 0.45} ${top - 6})` });
      t.textContent = tokText(trace.tokens[j]);
      svg.append(t);
    }
  }
  // outline the selected row
  svg.append(svgEl("rect", { x: left - 1, y: top + query * cell - 1, width: (query + 1) * cell + 1, height: cell + 1, class: "outline" }));
}

function onHeatClick(e) {
  const g = e.target.closest?.("g.row");
  if (!g) return;
  query = Number(g.dataset.row);
  renderAll();
}

// ---- one row, step by step ----------------------------------------------------

function renderRow() {
  const body = $("score-rows");
  body.replaceChildren();
  const q = trace.tokens[query];
  const title = $("row-title");
  title.replaceChildren("The token ", chip(q.bytes, "c0"), ` (position ${query + 1}) in layer ${layer + 1}, head ${head + 1}, looks back at ${query + 1} token${query ? "s" : ""}:`);
  const scores = Array.from({ length: query + 1 }, (_, j) => s(layer, head, query, j));
  const maxScore = Math.max(...scores);
  const exps = scores.map((x) => Math.exp(x - maxScore));       // shifted by the max so e^x cannot overflow; the ratios are the same
  const total = exps.reduce((a, b) => a + b, 0);
  let wsum = 0;
  scores.forEach((sc, j) => {
    const weight = exps[j] / total;
    wsum += weight;
    const tr = document.createElement("tr");
    if (j === query) tr.className = "self";
    const c0 = document.createElement("td");
    c0.append(chip(trace.tokens[j].bytes, "c2"));
    c0.append(document.createTextNode(j === query ? "  (itself)" : ""));
    const cells = [sc.toFixed(2), exps[j].toFixed(3), weight.toFixed(3)];
    tr.append(c0, ...cells.map((t) => { const td = document.createElement("td"); td.textContent = t; return td; }));
    const bar = document.createElement("td");
    bar.className = "barcell";
    const fill = document.createElement("span");
    fill.style.width = `${(weight * 100).toFixed(1)}%`;
    bar.append(fill);
    tr.append(bar);
    body.append(tr);
  });
  $("exp-total").textContent = total.toFixed(3);
  $("w-total").textContent = wsum.toFixed(3);
  // the e^score column is shown relative to the biggest score, which leaves every weight unchanged
  $("exp-total").title = "Each e^score is divided by e^(largest score) first, so big scores cannot overflow. The weights are identical.";
}

// ---- what each head does ------------------------------------------------------

function headSummary(l, h) {
  const T = trace.tokens.length;
  let n = 0, self = 0, prev = 0, first = 0, other = 0, even = 0;
  for (let i = SKIP; i < T; i++) {
    const a = w(l, h, i, i), p = w(l, h, i, i - 1), f = w(l, h, i, 0);
    self += a; prev += p; first += f; other += Math.max(0, 1 - a - p - f); even += 1 / (i + 1);
    n++;
  }
  return n ? { self: self / n, prev: prev / n, first: first / n, other: other / n, even: even / n, n } : null;
}

function renderHeads() {
  const body = $("head-rows");
  body.replaceChildren();
  let evenAvg = null;
  for (let l = 0; l < trace.n_layer; l++) {
    for (let h = 0; h < trace.n_head; h++) {
      const sum = headSummary(l, h);
      const tr = document.createElement("tr");
      if (l === layer && h === head) tr.className = "current";
      const name = document.createElement("td");
      name.textContent = `Layer ${l + 1}, head ${h + 1}`;
      tr.append(name);
      for (const key of ["self", "prev", "first", "other"]) {
        const td = document.createElement("td");
        td.textContent = sum ? `${(sum[key] * 100).toFixed(0)}%` : "-";
        tr.append(td);
      }
      body.append(tr);
      if (sum) evenAvg = sum.even;
    }
  }
  $("even-note").textContent = evenAvg === null
    ? "Type a longer text (at least five tokens) to see this table."
    : `For comparison: a token that spread its attention evenly would give each visible token about ${(evenAvg * 100).toFixed(0)}% on average, so anything much bigger than that is a real preference.`;
}

function publish() {
  const T = trace ? trace.tokens.length : 0;
  const rowSum = trace ? Array.from({ length: query + 1 }, (_, j) => w(layer, head, query, j)).reduce((a, b) => a + b, 0) : null;
  const summary = trace ? Array.from({ length: trace.n_layer * trace.n_head }, (_, k) => headSummary(Math.floor(k / trace.n_head), k % trace.n_head)) : [];
  document.body.dataset.book = JSON.stringify({ trained, which: which(), T, layer, head, query, rowSum, truncated: trace?.truncated ?? false, summary });
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
    trainedSession.free();
    trainedSession = new Session(sampleText, JSON.stringify(CONFIG));
    trained = false;
  }
  const t0 = performance.now();
  let done = 0, loss = 0;
  while (done < total) {
    const sliceStart = performance.now();
    while (done < total && performance.now() - sliceStart < 40) {
      const n = Math.min(2, total - done);
      loss = trainedSession.step(n);
      done += n;
    }
    bar.value = done / total;
    setStatus(`Training… step ${done} of ${total} (loss ${loss.toFixed(2)}, ${(done / ((performance.now() - t0) / 1000)).toFixed(0)} steps/s)`);
    await nextPaint();
  }
  bar.hidden = true;
  trained = true;
  const radio = document.querySelector('input[name="which"][value="trained"]');
  radio.disabled = false;
  radio.checked = true;
  $("trained-note").textContent = `(${total} steps, loss ${loss.toFixed(2)})`;
  $("btn-train").disabled = false;
  $("btn-train").textContent = "Train again";
  $("steps").disabled = false;
  setStatus(`Done: ${total} steps in ${((performance.now() - t0) / 1000).toFixed(1)} s. Switch between Untrained and Trained to compare.`);
  refresh();
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
    trainedSession = new Session(text, JSON.stringify(CONFIG));
    // Keep a frozen copy of the starting model so "Untrained" stays available after training.
    untrained = Session.from_model(trainedSession.export_model(), trainedSession.export_tokenizer(), undefined);
    $("model-line").textContent = `Model: ${trainedSession.num_params().toLocaleString()} numbers; ${CONFIG.n_layer} layers with ${CONFIG.n_head} attention heads each; window of ${CONFIG.ctx} tokens.`;
    $("text").addEventListener("input", refresh);
    $("layer").addEventListener("change", () => { layer = Number($("layer").value); renderAll(); });
    $("head").addEventListener("change", () => { head = Number($("head").value); renderAll(); });
    document.querySelectorAll('input[name="which"]').forEach((r) => r.addEventListener("change", refresh));
    $("heat").addEventListener("click", onHeatClick);
    $("btn-train").addEventListener("click", train);
    $("btn-train").disabled = false;
    setStatus("Ready. This is a random model, so its attention is nearly even. Press Train, then compare.");
    query = Number.MAX_SAFE_INTEGER;   // start on the last token
    refresh();
  } catch (e) {
    setStatus("Could not start: " + (e.message ?? e) + ". Build with scripts/build-web.ps1 and serve the web folder.", true);
  }
}
main();
