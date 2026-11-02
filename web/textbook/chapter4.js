// Chapter 4 widgets: gradient descent on a toy, a live loss curve for the real model, and
// "how surprised was the model?" for any text at any point in training.
import init, { Session, predict_for_text } from "../pkg/tlm_wasm.js";
import { $, chip, label, fetchSample, nextPaint } from "./common.js";

const SVG = "http://www.w3.org/2000/svg";
const CONFIG = { vocab_size: 512, d_model: 32, n_layer: 2, n_head: 2, ctx: 64, batch: 16, steps: 100000, max_bytes: 1000000 };
const GUESSES = 8;
const GUESS_LOSS = Math.log(CONFIG.vocab_size);     // loss of a model that knows nothing
const LOSS_CAP = 6;                                 // loss at which a token is shaded fully

const svgEl = (name, attrs = {}) => {
  const el = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
};
const setStatus = (msg, error = false) => { $("status").textContent = msg; $("status").classList.toggle("error", error); };

// ---- the toy: gradient descent on loss(w) = (w - 3)^2 -------------------------

const TARGET = 3;
const toy = { w: -3, trail: [], steps: 0 };
const toyLoss = (w) => (w - TARGET) ** 2;
const toySlope = (w) => 2 * (w - TARGET);
const lr = () => Number($("lr").value);

function toyReset() {
  toy.w = Number($("w0").value);
  toy.trail = [toy.w];
  toy.steps = 0;
  drawToy();
}

function toyStep() {
  toy.w -= lr() * toySlope(toy.w);
  toy.trail.push(toy.w);
  toy.steps++;
  drawToy();
}

function drawToy() {
  const svg = $("toy");
  svg.replaceChildren();
  const W = 600, H = 220, padL = 34, padR = 12, padT = 12, padB = 26;
  const x0 = -4, x1 = 8, yMax = 50;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const sx = (w) => padL + ((Math.max(x0, Math.min(x1, w)) - x0) / (x1 - x0)) * (W - padL - padR);
  const sy = (l) => H - padB - (Math.min(yMax, l) / yMax) * (H - padT - padB);
  svg.append(svgEl("rect", { x: padL, y: padT, width: W - padL - padR, height: H - padT - padB, class: "frame" }));
  for (let w = x0; w <= x1; w += 2) {
    const t = svgEl("text", { x: sx(w), y: H - 8, "text-anchor": "middle", class: "tick" });
    t.textContent = String(w);
    svg.append(t);
  }
  const yl = svgEl("text", { x: 4, y: padT + 10, class: "tick" });
  yl.textContent = "loss";
  svg.append(yl);
  svg.append(svgEl("line", { x1: sx(TARGET), x2: sx(TARGET), y1: padT, y2: H - padB, class: "best" }));
  let d = "";
  for (let w = x0; w <= x1 + 1e-9; w += 0.1) d += `${d ? "L" : "M"}${sx(w).toFixed(1)} ${sy(toyLoss(w)).toFixed(1)}`;
  svg.append(svgEl("path", { d, class: "bowl" }));
  if (toy.trail.length > 1) {
    svg.append(svgEl("polyline", { points: toy.trail.map((w) => `${sx(w).toFixed(1)},${sy(toyLoss(w)).toFixed(1)}`).join(" "), class: "trail" }));
  }
  const ball = svgEl("circle", { cx: sx(toy.w), cy: sy(toyLoss(toy.w)), r: 7, class: "ball" });
  const tip = svgEl("title");
  tip.textContent = `w = ${toy.w.toFixed(3)}, loss ${toyLoss(toy.w).toFixed(3)}`;
  ball.append(tip);
  svg.append(ball);

  const off = toy.w < x0 || toy.w > x1;
  const l = toyLoss(toy.w);
  const diverged = !Number.isFinite(l) || l > 1e6;
  $("toy-line").textContent = toy.steps === 0
    ? `Start: w = ${toy.w.toFixed(2)}, loss ${l.toFixed(2)}, slope ${toySlope(toy.w).toFixed(2)}. Press "Take a step".`
    : `Step ${toy.steps}: w = ${diverged ? "huge" : toy.w.toFixed(3)}, loss ${diverged ? "enormous" : l.toFixed(3)}${off ? " (off the chart)" : ""}${diverged ? ". It has diverged: the learning rate is too big." : ""}`;
  $("lr-out").textContent = lr().toFixed(2);
  $("w0-out").textContent = Number($("w0").value).toFixed(1);
  publish();
}

function toyRun() {
  for (let i = 0; i < 20; i++) {
    toy.w -= lr() * toySlope(toy.w);
    toy.trail.push(toy.w);
    toy.steps++;
  }
  drawToy();
}

// ---- training the real model --------------------------------------------------

let sampleText = "";
let session = null;               // the live session being trained
let total = 0;
let trainPts = [];                // [step, loss] for every call of step()
let valPts = [];                  // [step, held-out loss]
let snaps = [];                   // [{ step, bytes }]  model weights at chosen moments
let snapIdx = 0;
let view = null;                  // a session holding the model at the chosen snapshot
let trained = false;

function checkpoints(total) {
  const set = new Set([0, total]);
  for (let s = 10; s < total; s = Math.round(s * 3.2)) set.add(s);
  const every = Math.max(25, Math.round(total / 30));
  const vals = new Set(set);
  for (let s = every; s < total; s += every) vals.add(s);
  return { snaps: [...set].sort((a, b) => a - b), vals: [...vals].sort((a, b) => a - b) };
}

async function train() {
  total = Number($("steps").value);
  $("btn-train").disabled = true;
  $("steps").disabled = true;
  $("snap").disabled = true;
  const bar = $("progress");
  bar.hidden = false;
  bar.value = 0;
  setStatus("Starting a fresh model…");
  await nextPaint();
  session.free();
  session = new Session(sampleText, JSON.stringify(CONFIG));
  trained = false;
  trainPts = [];
  valPts = [[0, session.validate()]];
  snaps = [{ step: 0, bytes: session.export_model() }];
  snapIdx = 0;
  const cp = checkpoints(total);
  const t0 = performance.now();
  let done = 0, loss = 0;
  for (const target of cp.vals.filter((s) => s > 0)) {
    while (done < target) {
      const sliceStart = performance.now();
      while (done < target && performance.now() - sliceStart < 40) {
        const n = Math.min(2, target - done);
        loss = session.step(n);
        done += n;
        trainPts.push([done, loss]);
      }
      bar.value = done / total;
      setStatus(`Training… step ${done} of ${total} (loss ${loss.toFixed(2)}, ${(done / ((performance.now() - t0) / 1000)).toFixed(0)} steps/s)`);
      drawCurve();
      await nextPaint();
    }
    valPts.push([done, session.validate()]);
    if (cp.snaps.includes(done)) snaps.push({ step: done, bytes: session.export_model() });
  }
  bar.hidden = true;
  trained = true;
  $("btn-train").disabled = false;
  $("btn-train").textContent = "Train again";
  $("steps").disabled = false;
  const slider = $("snap");
  slider.max = String(snaps.length - 1);
  slider.disabled = false;
  const secs = (performance.now() - t0) / 1000;
  setStatus(`Done: ${total} steps in ${secs.toFixed(1)} s. Held-out loss went from ${valPts[0][1].toFixed(2)} to ${valPts[valPts.length - 1][1].toFixed(2)}. Drag the slider in section 4 to compare.`);
  setSnapshot(snaps.length - 1);
}

function drawCurve() {
  const svg = $("curve");
  svg.replaceChildren();
  const W = 600, H = 260, padL = 36, padR = 12, padT = 12, padB = 28;
  const xMax = Math.max(total, 1), yMax = 7;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const sx = (s) => padL + (s / xMax) * (W - padL - padR);
  const sy = (l) => H - padB - (Math.max(0, Math.min(yMax, l)) / yMax) * (H - padT - padB);
  svg.append(svgEl("rect", { x: padL, y: padT, width: W - padL - padR, height: H - padT - padB, class: "frame" }));
  for (let l = 0; l <= 6; l += 2) {
    svg.append(svgEl("line", { x1: padL, x2: W - padR, y1: sy(l), y2: sy(l), class: "gridl" }));
    const t = svgEl("text", { x: padL - 6, y: sy(l) + 4, "text-anchor": "end", class: "tick" });
    t.textContent = String(l);
    svg.append(t);
  }
  for (let i = 0; i <= 4; i++) {
    const t = svgEl("text", { x: sx((xMax * i) / 4), y: H - 8, "text-anchor": "middle", class: "tick" });
    t.textContent = String(Math.round((xMax * i) / 4));
    svg.append(t);
  }
  svg.append(svgEl("line", { x1: padL, x2: W - padR, y1: sy(GUESS_LOSS), y2: sy(GUESS_LOSS), class: "floor" }));
  if (trainPts.length) {
    svg.append(svgEl("polyline", { points: trainPts.map(([s, l]) => `${sx(s).toFixed(1)},${sy(l).toFixed(1)}`).join(" "), class: "train" }));
  }
  if (valPts.length) {
    svg.append(svgEl("polyline", { points: valPts.map(([s, l]) => `${sx(s).toFixed(1)},${sy(l).toFixed(1)}`).join(" "), class: "val" }));
    for (const [s, l] of valPts) {
      const c = svgEl("circle", { cx: sx(s), cy: sy(l), r: 3, class: "valpt" });
      const t = svgEl("title");
      t.textContent = `step ${s}: held-out loss ${l.toFixed(3)}`;
      c.append(t);
      svg.append(c);
    }
  }
  if (trained && snaps[snapIdx]) {
    const s = snaps[snapIdx].step;
    svg.append(svgEl("line", { x1: sx(s), x2: sx(s), y1: padT, y2: H - padB, class: "mark" }));
  }
}

// ---- looking at the model at one moment ---------------------------------------

let pred = null;
let selected = 0;

function setSnapshot(i) {
  snapIdx = i;
  if (view) view.free();
  view = Session.from_model(snaps[i].bytes, session.export_tokenizer(), undefined);
  $("snap").value = String(i);
  $("snap-out").textContent = snaps[i].step.toLocaleString();
  drawCurve();
  refresh();
}

function refresh() {
  const model = view ?? session;
  try {
    pred = JSON.parse(predict_for_text(model, $("text").value, GUESSES));
  } catch (e) {
    pred = null;
    $("surprise").replaceChildren();
    $("guess-list").replaceChildren();
    $("guess-title").textContent = "";
    $("mean-line").textContent = String(e.message ?? e);
    $("trunc-note").hidden = true;
    publish();
    return;
  }
  const T = pred.tokens.length;
  selected = Math.max(0, Math.min(T - 1, selected === Number.MAX_SAFE_INTEGER ? T - 2 : selected));
  $("trunc-note").hidden = !pred.truncated;
  $("trunc-note").textContent = pred.truncated ? `Your text is longer than the model's ${T}-token window, so only the first ${T} tokens are shown.` : "";
  $("mean-line").textContent = `Average loss on this text: ${pred.mean_loss.toFixed(2)} (guessing would score ${GUESS_LOSS.toFixed(2)}).`;
  renderTokens();
  renderGuesses();
  publish();
}

const tokenLoss = (i) => (i === 0 ? null : -Math.log(Math.max(1e-30, pred.positions[i - 1].p_actual)));

function renderTokens() {
  const box = $("surprise");
  box.replaceChildren();
  pred.tokens.forEach((t, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.i = String(i);
    const l = tokenLoss(i);
    b.style.setProperty("--s", l === null ? 0 : Math.min(1, l / LOSS_CAP).toFixed(3));
    b.textContent = label(t.bytes).text;
    b.title = l === null ? "first token: there is nothing before it to predict it from" : `loss ${l.toFixed(2)} (the model gave it ${(Math.exp(-l) * 100).toFixed(1)}%)`;
    b.setAttribute("aria-pressed", String(i === selected));
    box.append(b);
  });
}

function renderGuesses() {
  const list = $("guess-list");
  list.replaceChildren();
  const pos = pred.positions[selected];
  const next = pred.tokens[selected + 1];
  const title = $("guess-title");
  title.replaceChildren("After ", chip(pred.tokens[selected].bytes, "c0"), next ? " the real next token was " : " (the last token, so there is no real next token) the model guesses:");
  if (next) title.append(chip(next.bytes, "c1"), ". The model's top guesses:");
  const addRow = (bytes, prob, tag, isActual) => {
    const li = document.createElement("li");
    if (isActual) li.className = "actual";
    const first = document.createElement("span");
    first.append(chip(bytes, "c2"));
    if (tag) {
      const t = document.createElement("span");
      t.className = "tag";
      t.textContent = tag;
      first.append(t);
    }
    li.append(first);
    const bar = document.createElement("span");
    bar.className = "sim";
    const fill = document.createElement("span");
    fill.style.width = `${Math.min(100, prob * 100)}%`;
    bar.append(fill);
    const num = document.createElement("span");
    num.className = "num";
    num.textContent = prob >= 0.001 ? `${(prob * 100).toFixed(1)}%` : "<0.1%";
    li.append(bar, num);
    list.append(li);
  };
  let found = false;
  for (const g of pos.top) {
    const isActual = next && g.id === next.id;
    found ||= !!isActual;
    addRow(g.bytes, g.prob, isActual ? "← real" : "", isActual);
  }
  if (next && !found) addRow(next.bytes, pos.p_actual, "← real", true);
}

function onTokenClick(e) {
  const b = e.target.closest?.("button[data-i]");
  if (!b) return;
  selected = Number(b.dataset.i);
  renderTokens();
  renderGuesses();
  publish();
}

function publish() {
  document.body.dataset.book = JSON.stringify({
    trained, total, trainPts: trainPts.length, valPts: valPts.map(([s, l]) => [s, +l.toFixed(4)]),
    snaps: snaps.map((s) => s.step), snapIdx,
    meanLoss: pred ? +pred.mean_loss.toFixed(4) : null, T: pred ? pred.tokens.length : 0, selected,
    guesses: pred ? pred.positions[selected].top.length : 0,
    probSum: pred ? +pred.positions[selected].top.reduce((a, g) => a + g.prob, 0).toFixed(4) : null,
    params: session ? session.num_params() : null,
    toy: { w: toy.w, steps: toy.steps, lr: lr() },
  });
}

// ---- start up -----------------------------------------------------------------

async function main() {
  toyReset();
  $("lr").addEventListener("input", drawToy);
  $("w0").addEventListener("input", toyReset);
  $("toy-step").addEventListener("click", toyStep);
  $("toy-run").addEventListener("click", toyRun);
  $("toy-reset").addEventListener("click", toyReset);
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
    valPts = [[0, session.validate()]];
    snaps = [{ step: 0, bytes: session.export_model() }];
    total = Number($("steps").value);
    $("model-line").textContent = `Model: ${session.num_params().toLocaleString()} numbers; ${CONFIG.n_layer} layers; window of ${CONFIG.ctx} tokens; vocabulary of ${CONFIG.vocab_size}.`;
    $("text").addEventListener("input", refresh);
    $("surprise").addEventListener("click", onTokenClick);
    $("snap").addEventListener("input", () => setSnapshot(Number($("snap").value)));
    $("btn-train").addEventListener("click", train);
    $("btn-train").disabled = false;
    drawCurve();
    selected = Number.MAX_SAFE_INTEGER;   // start on the second-to-last token
    refresh();
    setStatus("Ready. Right now the model is random: every guess is about as bad as it can be. Press Train.");
  } catch (e) {
    setStatus("Could not start: " + (e.message ?? e) + ". Build with scripts/build-web.ps1 and serve the web folder.", true);
  }
}
main();
