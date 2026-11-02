// Chapter 2 widgets: train a tiny model in the browser, keep snapshots of its embedding table,
// and look at one token's vector, its nearest neighbours, and a 2-D map of the whole table.
import init, { Session, EmbeddingView } from "../pkg/tlm_wasm.js";
import { $, chip, label, tokenString, fetchSample, nextPaint } from "./common.js";

const SNAPSHOTS = 10;                 // snapshots kept during training (plus step 0)
const PROBES = [" the", " a", " was", " were", " dog", " cat", " girl", " happy", ".", ","];
const NEIGHBORS = 8;
const MAP = { w: 600, h: 420, pad: 22 };
const SVG = "http://www.w3.org/2000/svg";

let vocab = [];            // token id -> bytes
let strings = [];          // token id -> text or null
let byText = new Map();    // text -> token id
let snaps = [];            // [{ step, loss, view }]
let maps = [];             // per-snapshot map coordinates (all on the final table's axes once trained)
let extent = null;         // map extent from the final snapshot
let selected = -1;
let trained = false;
let session = null;
let sampleText = "";

const CONFIG = { vocab_size: 512, d_model: 32, n_layer: 2, n_head: 2, ctx: 64, batch: 16, steps: 100000, max_bytes: 1000000 };

const svgEl = (name, attrs = {}) => {
  const el = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
};

function kindOf(id) {
  const s = strings[id];
  if (s === null) return "raw";
  if (/^ [A-Za-z]+$/.test(s)) return "word";
  if (/[A-Za-z]/.test(s)) return "frag";
  return "sym";
}

function setStatus(msg, error = false) {
  const el = $("status");
  el.textContent = msg;
  el.classList.toggle("error", error);
}

// ---- token choice -------------------------------------------------------------

/** Find the token whose text is exactly `text` (also try one leading space). */
function findToken(text) {
  if (byText.has(text)) return byText.get(text);
  if (text && byText.has(" " + text)) return byText.get(" " + text);
  return -1;
}

function selectToken(id, fromInput = false) {
  selected = id;
  // Type-able text when the token is plain printable ASCII; otherwise its readable label (· for space, \xNN for raw bytes).
  if (!fromInput) $("token-input").value = strings[id] !== null && /^[\x20-\x7e]+$/.test(strings[id]) ? strings[id] : label(vocab[id]).text;
  $("token-note").textContent = "";
  render();
}

function onTokenInput() {
  const text = $("token-input").value;
  if (text === "") {
    $("token-note").textContent = "";
    return;
  }
  const id = findToken(text);
  if (id < 0) {
    $("token-note").textContent = `No token is exactly "${text}". Tokens are pieces of text; most words start with a space, so try " ${text.trim()}" (type a space first) or pick one of the examples.`;
    return;
  }
  selectToken(id, true);
}

// ---- drawing ------------------------------------------------------------------

function drawVector(view) {
  const svg = $("vec");
  svg.replaceChildren();
  const v = view.vector(selected);
  const n = v.length;
  const W = 600, H = 120, mid = H / 2;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const scale = vectorScale / 1.0;
  const bw = W / n;
  svg.append(svgEl("line", { x1: 0, x2: W, y1: mid, y2: mid, class: "axis" }));
  v.forEach((x, i) => {
    const h = Math.min(mid - 2, (Math.abs(x) / scale) * (mid - 2));
    const r = svgEl("rect", { x: i * bw + 1, width: Math.max(1, bw - 2), y: x >= 0 ? mid - h : mid, height: h, class: x >= 0 ? "pos" : "neg" });
    const t = svgEl("title");
    t.textContent = `dimension ${i + 1}: ${x.toFixed(3)}`;
    r.append(t);
    svg.append(r);
  });
}

let vectorScale = 1;

function neighborList(listEl, view) {
  listEl.replaceChildren();
  const items = JSON.parse(view.neighbors(selected, NEIGHBORS));
  for (const it of items) {
    const li = document.createElement("li");
    li.append(chip(vocab[it.id], "c2"));
    const bar = document.createElement("span");
    bar.className = "sim";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(0, Math.min(1, it.similarity)) * 100}%`;
    bar.append(fill);
    const num = document.createElement("span");
    num.className = "num";
    num.textContent = it.similarity.toFixed(2);
    li.append(bar, num);
    li.title = "click to select";
    li.tabIndex = 0;
    li.addEventListener("click", () => selectToken(it.id));
    li.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), selectToken(it.id)));
    listEl.append(li);
  }
  return items;
}

function mapPoint(i, xy) {
  const { minX, maxX, minY, maxY } = extent;
  const sx = (MAP.w - 2 * MAP.pad) / (maxX - minX || 1);
  const sy = (MAP.h - 2 * MAP.pad) / (maxY - minY || 1);
  // Outliers beyond the trimmed range are pinned to the frame edge instead of vanishing.
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  return [
    clamp(MAP.pad + (xy[2 * i] - minX) * sx, 4, MAP.w - 4),
    clamp(MAP.h - MAP.pad - (xy[2 * i + 1] - minY) * sy, 4, MAP.h - 4),
  ];
}

function drawMap(idx, neighborIds) {
  const svg = $("map");
  svg.replaceChildren();
  const xy = maps[idx];
  const near = new Set(neighborIds);
  const frame = svgEl("rect", { x: 0.5, y: 0.5, width: MAP.w - 1, height: MAP.h - 1, class: "frame", rx: 8 });
  svg.append(frame);
  for (let i = 0; i < vocab.length; i++) {
    const [x, y] = mapPoint(i, xy);
    const c = svgEl("circle", { cx: x.toFixed(1), cy: y.toFixed(1), r: i === selected ? 6 : near.has(i) ? 4.5 : 3, class: `pt k-${kindOf(i)}${i === selected ? " sel" : ""}${near.has(i) ? " nn" : ""}`, "data-id": i });
    svg.append(c);
  }
  // labels: the selected token, its neighbours and the example tokens
  const labelled = new Set([selected, ...neighborIds, ...probeIds]);
  for (const i of labelled) {
    const [x, y] = mapPoint(i, xy);
    const t = svgEl("text", { x: (x + 7).toFixed(1), y: (y + 4).toFixed(1), class: i === selected ? "lab sel" : "lab" });
    t.textContent = label(vocab[i]).text;
    svg.append(t);
  }
  // the selected dot goes on top
  const top = svg.querySelector("circle.sel");
  if (top) svg.append(top);
}

let probeIds = [];

function render() {
  if (selected < 0 || snaps.length === 0) return;
  const idx = Number($("snap").value);
  const snap = snaps[idx];
  const view = snap.view;
  $("snap-out").textContent = idx === 0 ? "step 0 (random)" : `step ${snap.step}` + (snap.loss ? `, loss ${snap.loss.toFixed(2)}` : "");
  $("nn-now-title").textContent = idx === 0 ? "At the start (same as left)" : `After ${snap.step} steps`;
  drawVector(view);
  neighborList($("nn-before"), snaps[0].view);
  const now = neighborList($("nn-now"), view);
  drawMap(idx, now.map((n) => n.id));
  publishState(idx, now);
}

function publishState(idx, now) {
  const snap = snaps[idx];
  document.body.dataset.book = JSON.stringify({
    trained, snapshots: snaps.length, idx, step: snap.step, loss: snap.loss, selected,
    selectedText: strings[selected], neighbors: now.map((n) => n.id), top: now[0]?.similarity ?? null,
    vocab: vocab.length, dim: snap.view.dim(),
  });
}

// ---- training -----------------------------------------------------------------

async function train() {
  const total = Number($("steps").value);
  const every = Math.max(1, Math.floor(total / SNAPSHOTS));
  $("btn-train").disabled = true;
  $("steps").disabled = true;
  $("snap").disabled = true;
  const bar = $("progress");
  bar.hidden = false;
  bar.value = 0;
  if (trained) {
    // "Train again" starts over from a fresh model (same seed, so the same random start as snapshot 0).
    setStatus("Starting a fresh model…");
    await nextPaint();
    session.free();
    session = new Session(sampleText, JSON.stringify(CONFIG));
  }
  snaps = snaps.slice(0, 1);
  maps = [Array.from(snaps[0].view.map())];
  trained = false;
  const t0 = performance.now();
  let done = 0;
  let loss = 0;
  while (done < total) {
    // Work in slices of about 40 ms so the page stays responsive.
    const sliceStart = performance.now();
    while (done < total && performance.now() - sliceStart < 40) {
      const n = Math.min(2, every - (done % every), total - done);
      loss = session.step(n);
      done += n;
      if (done % every === 0 || done === total) snaps.push({ step: done, loss, view: new EmbeddingView(session) });
    }
    bar.value = done / total;
    const secs = (performance.now() - t0) / 1000;
    setStatus(`Training… step ${done} of ${total} (loss ${loss.toFixed(2)}, ${(done / secs).toFixed(0)} steps/s)`);
    await nextPaint();
  }
  bar.hidden = true;
  finishTraining(total, (performance.now() - t0) / 1000);
}

function finishTraining(total, secs) {
  const final = snaps[snaps.length - 1].view;
  // Every snapshot is drawn on the FINAL table's map axes, so the picture holds still while scrubbing.
  maps = snaps.map((s) => Array.from(s.view.map_with(final)));
  const all = maps[maps.length - 1];
  extent = bounds(all, 0.01);
  trained = true;
  const [e1, e2] = Array.from(final.explained_variance());
  $("map-var").textContent = `The two map axes keep ${(100 * (e1 + e2)).toFixed(0)}% of the variation in the final table (${(100 * e1).toFixed(0)}% and ${(100 * e2).toFixed(0)}%); the other directions are not shown, and the few most extreme tokens are pinned to the frame edge. At step 0 the table is random, so its dots form an unstructured cloud around the centre.`;
  vectorScale = tableScale(final);
  const slider = $("snap");
  slider.max = String(snaps.length - 1);
  slider.value = String(snaps.length - 1);
  slider.disabled = false;
  $("btn-train").disabled = false;
  $("btn-train").textContent = "Train again";
  $("steps").disabled = false;
  setStatus(`Done: ${total} steps in ${secs.toFixed(1)} s. Drag the snapshot slider back to step 0 to see the table before training.`);
  render();
}

/** Axis range that ignores the most extreme `trim` fraction of points on each side, so one outlier cannot squash the map. */
function bounds(xy, trim) {
  const xs = [], ys = [];
  for (let i = 0; i < xy.length; i += 2) { xs.push(xy[i]); ys.push(xy[i + 1]); }
  const q = (a, p) => { const s = [...a].sort((u, v) => u - v); return s[Math.min(s.length - 1, Math.max(0, Math.round(p * (s.length - 1))))]; };
  let [minX, maxX, minY, maxY] = [q(xs, trim), q(xs, 1 - trim), q(ys, trim), q(ys, 1 - trim)];
  const padX = (maxX - minX) * 0.05 || 1, padY = (maxY - minY) * 0.05 || 1;
  return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
}

/** Largest typical vector entry of a table (99th percentile of |x|), so bars are comparable across snapshots. */
function tableScale(view) {
  const vals = [];
  for (let id = 0; id < view.vocab_size(); id++) for (const x of view.vector(id)) vals.push(Math.abs(x));
  vals.sort((a, b) => a - b);
  return vals[Math.floor(vals.length * 0.99)] || 1;
}

// ---- start up -----------------------------------------------------------------

function onMapEvent(e) {
  const t = e.target.closest?.("circle.pt");
  const tip = $("map-tip");
  if (!t) { tip.hidden = true; return; }
  const id = Number(t.dataset.id);
  if (e.type === "click") { selectToken(id); return; }
  const box = $("map").getBoundingClientRect();
  tip.textContent = `${label(vocab[id]).text}  (token ${id})`;
  tip.hidden = false;
  tip.style.left = `${e.clientX - box.left + 12}px`;
  tip.style.top = `${e.clientY - box.top + 12}px`;
}

async function main() {
  try {
    await init();
    setStatus("Loading the story sample…");
    const text = await fetchSample();
    if (!text) {
      setStatus("The story sample (web/data/tinystories-8mb.txt) was not found. See the README for how to create it.", true);
      return;
    }
    setStatus("Preparing a tiny model (learning the vocabulary from the stories)…");
    await nextPaint();
    sampleText = text;
    session = new Session(text, JSON.stringify(CONFIG));
    const view0 = new EmbeddingView(session);
    snaps = [{ step: 0, loss: 0, view: view0 }];
    vocab = JSON.parse(view0.vocab_json());
    strings = vocab.map(tokenString);
    strings.forEach((s, id) => { if (s !== null && !byText.has(s)) byText.set(s, id); });

    const list = $("token-list");
    strings.forEach((s) => { if (s !== null && s.trim() !== "" && /^[\x20-\x7e]+$/.test(s)) { const o = document.createElement("option"); o.value = s; list.append(o); } });
    probeIds = PROBES.map((p) => byText.get(p)).filter((id) => id !== undefined);
    for (const id of probeIds) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip-btn";
      b.append(chip(vocab[id], "c2"));
      b.setAttribute("aria-label", `select token ${strings[id]}`);
      b.addEventListener("click", () => selectToken(id));
      $("probes").append(b);
    }

    vectorScale = tableScale(view0) * 1.5;
    // Before training there is no "final" table, so step 0 uses its own axes.
    maps = [Array.from(view0.map())];
    extent = bounds(maps[0], 0.01);
    selected = byText.get(" the") ?? probeIds[0] ?? 256;
    $("token-input").value = strings[selected] ?? "";
    $("dim-out").textContent = String(view0.dim());
    $("model-line").textContent = `Model: ${session.num_params().toLocaleString()} numbers in total; the embedding table alone is ${view0.vocab_size()} tokens × ${view0.dim()} numbers = ${(view0.vocab_size() * view0.dim()).toLocaleString()}.`;
    $("token-input").addEventListener("input", onTokenInput);
    $("snap").addEventListener("input", render);
    $("btn-train").addEventListener("click", train);
    const map = $("map");
    ["mousemove", "click", "mouseleave"].forEach((t) => map.addEventListener(t, onMapEvent));
    $("btn-train").disabled = false;
    setStatus("Ready. The table below is still random. Press Train.");
    render();
  } catch (e) {
    setStatus("Could not start: " + (e.message ?? e) + ". Build with scripts/build-web.ps1 and serve the web folder.", true);
  }
}
main();
