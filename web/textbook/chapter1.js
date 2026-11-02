// Chapter 1 widgets: a tokenizer playground over the real BPE engine (compiled to WebAssembly).
import init, { TokenizerLab } from "../pkg/tlm_wasm.js";
import { $, label, chip } from "./common.js";

const MAX_MERGES = 512;       // the slider's range; the medium model uses 768 merges
const FALLBACK_TEXT =
  "Once upon a time, a little girl named Lily found a shiny red ball in the garden. She played with the ball all day. " +
  "The dog and the cat saw the ball too, and they wanted to play. One day Tom and his mom went to the park. ";

let lab = null;
let merges = [];

async function loadSampleText() {
  try {
    const r = await fetch("../data/tinystories-8mb.txt");
    if (!r.ok) throw new Error(String(r.status));
    return { text: await r.text(), real: true };
  } catch {
    return { text: FALLBACK_TEXT.repeat(400), real: false };
  }
}

function renderRecipe(k) {
  const body = $("recipe");
  body.replaceChildren();
  const rows = merges.length;
  for (let i = 0; i < rows; i++) {
    const m = merges[i];
    const tr = document.createElement("tr");
    if (i >= k) tr.className = "future";
    else if (i === k - 1) tr.className = "current";
    const cells = [String(i + 1), String(m.id), `${label(m.left).text} + ${label(m.right).text}`, label([...m.left, ...m.right]).text];
    for (const c of cells) {
      const td = document.createElement("td");
      td.textContent = c;
      tr.append(td);
    }
    body.append(tr);
  }
  // Keep the newest merge in view inside the scrolling table (scroll the box, not the page).
  const wrap = body.closest(".table-wrap");
  const cur = body.querySelector("tr.current");
  if (wrap && cur) wrap.scrollTop = Math.max(0, cur.offsetTop - wrap.clientHeight / 2);
  else if (wrap) wrap.scrollTop = 0;
}

function render() {
  const k = Number($("k").value);
  lab.set_merges(k);
  $("k-out").textContent = String(k);
  $("s-vocab").textContent = String(lab.vocab_size());

  const text = $("text").value;
  const chips = $("chips");
  chips.replaceChildren();
  let tokens = [];
  try {
    tokens = JSON.parse(lab.encode(text));
  } catch (e) {
    $("roundtrip").textContent = String(e.message ?? e);
  }
  tokens.forEach((t, i) => chips.append(chip(t.bytes, "c" + (i % 3))));
  if (tokens.length === 0) {
    const s = document.createElement("span");
    s.className = "empty";
    s.textContent = "Type something above.";
    chips.append(s);
  }

  const nBytes = new TextEncoder().encode(text).length;
  $("s-bytes").textContent = String(nBytes);
  $("s-tokens").textContent = String(tokens.length);
  $("s-ratio").textContent = tokens.length ? (nBytes / tokens.length).toFixed(2) : "0";
  $("ids").textContent = tokens.map((t) => t.id).join(" ");

  if (tokens.length) {
    const back = lab.decode(Uint32Array.from(tokens.map((t) => t.id)));
    $("roundtrip").textContent = back === text ? "Decoding these numbers gives back exactly your text." : "Round trip mismatch (this should never happen).";
  } else {
    $("roundtrip").textContent = "";
  }

  const latest = $("latest");
  latest.replaceChildren();
  if (k === 0) {
    latest.textContent = "Move the slider to learn the first merge.";
  } else {
    const m = merges[k - 1];
    latest.append(`Merge ${k}: `, chip(m.left, "c0"), " + ", chip(m.right, "c1"), " → ", chip([...m.left, ...m.right], "c2"), ` (token ${m.id})`);
  }
  renderRecipe(k);

  document.body.dataset.book = JSON.stringify({ k, vocab: lab.vocab_size(), bytes: nBytes, tokens: tokens.length });
}

async function main() {
  const status = $("lab-status");
  try {
    await init();
    const { text, real } = await loadSampleText();
    const t0 = performance.now();
    lab = new TokenizerLab(text, MAX_MERGES);
    const learned = lab.learned_merges();
    merges = JSON.parse(lab.merges(learned));
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    status.textContent = `Ready. Learned ${learned} merges from ${real ? "the TinyStories sample" : "a tiny built-in sample (the TinyStories file was not found)"} in ${secs} s.`;

    const slider = $("k");
    slider.max = String(learned);
    slider.disabled = false;
    slider.addEventListener("input", render);
    $("text").addEventListener("input", render);
    render();
  } catch (e) {
    status.classList.add("error");
    status.textContent = "Could not start the engine: " + (e.message ?? e) + ". Build it with scripts/build-web.ps1 and serve the web folder.";
  }
}
main();
