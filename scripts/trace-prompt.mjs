// Trace one prompt through the tiny transformer, step by step, printing real numbers.
// It re-implements the forward pass from crates/tlm-core/src/transformer.rs in plain JS.
//
// Usage:  node scripts/trace-prompt.mjs <run-dir> "<prompt>" [--top N] [--show N] [--qkv [--layer L] [--head H]]
// Example: node scripts/trace-prompt.mjs runs/demo "Once upon a time"

import { readFileSync } from "node:fs";
import { join } from "node:path";

// ---- args -------------------------------------------------------------------
const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 ? Number(args.splice(i, 2)[1]) : dflt;
};
const flag = (name) => { const i = args.indexOf(name); if (i >= 0) args.splice(i, 1); return i >= 0; };
const QKV = flag("--qkv"); // print the q, k, v vectors and raw scores of one head
const QKV_LAYER = opt("--layer", 0);
const QKV_HEAD = opt("--head", 0);
const TOP = opt("--top", 5);
const SHOW = opt("--show", 6); // how many numbers of a vector to print
const dir = args[0] ?? "runs/demo";
const prompt = args[1] ?? "Once upon a time";

// ---- load weights (safetensors, F32) ------------------------------------------
const buf = readFileSync(join(dir, "model.safetensors"));
const hlen = Number(buf.readBigUInt64LE(0));
const header = JSON.parse(buf.subarray(8, 8 + hlen).toString("utf8"));
const body = buf.subarray(8 + hlen);
const meta = header.__metadata__;
const W = {}; // name -> { rows, cols, data }
for (const [name, t] of Object.entries(header)) {
  if (name === "__metadata__") continue;
  const [s, e] = t.data_offsets;
  const data = new Float32Array(Uint8Array.prototype.slice.call(body, s, e).buffer);
  const [rows, cols] = t.shape.length === 1 ? [1, t.shape[0]] : t.shape;
  W[name] = { rows, cols, data };
}
const D = Number(meta.d_model), NH = Number(meta.n_head), NL = Number(meta.n_layer);
const CTX = Number(meta.ctx), V = Number(meta.vocab_size), DH = D / NH;

// ---- tokenizer (byte-level BPE, same rules as crates/tlm-core/src/bpe.rs) -----
const merges = JSON.parse(readFileSync(join(dir, "tokenizer.json"), "utf8")).merges;
const vocab = Array.from({ length: 256 }, (_, b) => [b]);
const rank = new Map();
merges.forEach(([a, b], k) => { vocab.push([...vocab[a], ...vocab[b]]); rank.set(a + "," + b, [k, 256 + k]); });

function pretokenize(bytes) {
  const chunks = []; let i = 0;
  while (i < bytes.length) {
    const start = i;
    if (bytes[i] === 10) i++;
    else { if (bytes[i] === 32) i++; while (i < bytes.length && bytes[i] !== 32 && bytes[i] !== 10) i++; }
    chunks.push(bytes.slice(start, i));
  }
  return chunks;
}
function encode(text) {
  const out = [];
  for (const chunk of pretokenize([...Buffer.from(text, "utf8")])) {
    let toks = [...chunk];
    for (;;) {
      let best = null;
      for (let i = 0; i + 1 < toks.length; i++) {
        const r = rank.get(toks[i] + "," + toks[i + 1]);
        if (r && (!best || r[0] < best[0])) best = r;
      }
      if (!best) break;
      const [a, b] = merges[best[0]], merged = [];
      for (let i = 0; i < toks.length; i++) {
        if (i + 1 < toks.length && toks[i] === a && toks[i + 1] === b) { merged.push(best[1]); i++; }
        else merged.push(toks[i]);
      }
      toks = merged;
    }
    out.push(...toks);
  }
  return out;
}
const show = (id) => JSON.stringify(Buffer.from(vocab[id]).toString("utf8"));

// ---- tiny matrix helpers (row-major Float32Array, like the Rust Matrix) --------
const mat = (rows, cols) => ({ rows, cols, data: new Float32Array(rows * cols) });
function matmul(a, b) {
  const o = mat(a.rows, b.cols);
  for (let i = 0; i < a.rows; i++)
    for (let k = 0; k < a.cols; k++) {
      const x = a.data[i * a.cols + k];
      for (let j = 0; j < b.cols; j++) o.data[i * b.cols + j] += x * b.data[k * b.cols + j];
    }
  return o;
}
const addBias = (m, b) => { for (let i = 0; i < m.rows; i++) for (let j = 0; j < m.cols; j++) m.data[i * m.cols + j] += b.data[j]; };
const add = (a, b) => { const o = mat(a.rows, a.cols); for (let i = 0; i < a.data.length; i++) o.data[i] = a.data[i] + b.data[i]; return o; };
function layerNorm(x, g, b) {
  const o = mat(x.rows, x.cols);
  for (let r = 0; r < x.rows; r++) {
    const row = x.data.subarray(r * x.cols, (r + 1) * x.cols);
    const mean = row.reduce((s, v) => s + v, 0) / x.cols;
    const variance = row.reduce((s, v) => s + (v - mean) ** 2, 0) / x.cols;
    const rs = 1 / Math.sqrt(variance + 1e-5);
    for (let j = 0; j < x.cols; j++) o.data[r * x.cols + j] = (row[j] - mean) * rs * g.data[j] + b.data[j];
  }
  return o;
}
const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x ** 3)));
function softmax(v) {
  const m = Math.max(...v), e = v.map((x) => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

// ---- printing helpers --------------------------------------------------------
const f = (x) => (x >= 0 ? " " : "") + x.toFixed(3);
const vec = (m, row) => {
  const r = Array.from(m.data.subarray(row * m.cols, row * m.cols + Math.min(SHOW, m.cols)), f).join(" ");
  const norm = Math.sqrt(m.data.subarray(row * m.cols, (row + 1) * m.cols).reduce((s, v) => s + v * v, 0));
  return `[${r} ...]  |len| ${norm.toFixed(2)}`;
};
const hr = (t) => console.log(`\n${"=".repeat(4)} ${t} ${"=".repeat(Math.max(4, 70 - t.length))}`);

// ---- run ---------------------------------------------------------------------
let ids = encode(prompt);
if (ids.length > CTX) { console.log(`(prompt is ${ids.length} tokens; keeping the last ${CTX})`); ids = ids.slice(-CTX); }
const T = ids.length;
console.log(`Model: ${dir}   d_model=${D} heads=${NH} (d_head ${DH}) layers=${NL} ctx=${CTX} vocab=${V}`);

hr("1. TOKENIZE: text -> token ids");
console.log(`prompt: ${JSON.stringify(prompt)}`);
ids.forEach((id, i) => console.log(`  pos ${i}: id ${String(id).padStart(3)}  ${show(id)}`));

hr("2. EMBED: look up a row of tok_emb (what) + a row of pos_emb (where), then add");
let x = mat(T, D);
for (let i = 0; i < T; i++)
  for (let j = 0; j < D; j++) x.data[i * D + j] = W.tok_emb.data[ids[i] * D + j] + W.pos_emb.data[i * D + j];
ids.forEach((id, i) => {
  console.log(`  pos ${i} ${show(id).padEnd(10)} tok_emb[${id}] + pos_emb[${i}] =`);
  console.log(`      ${vec(x, i)}`);
});

const trace = []; // keep attention per layer/head for the summary
for (let l = 0; l < NL; l++) {
  const p = (n) => W[`layers.${l}.${n}`];
  hr(`LAYER ${l}: attention block (x = x + Attn(LayerNorm(x)))`);
  const ln1 = layerNorm(x, p("ln1.gain"), p("ln1.shift"));
  const q = matmul(ln1, p("attn.wq"));
  const k =matmul(ln1, p("attn.wk")), v = matmul(ln1, p("attn.wv"));
  const cat = mat(T, D);
  const heads = [];
  for (let h = 0; h < NH; h++) {
    const A = [];
    for (let i = 0; i < T; i++) {
      const scores = [];
      for (let j = 0; j <= i; j++) {
        let s = 0;
        for (let c = 0; c < DH; c++) s += q.data[i * D + h * DH + c] * k.data[j * D + h * DH + c];
        scores.push(s / Math.sqrt(DH));
      }
      const a = softmax(scores);
      A.push(a);
      for (let j = 0; j <= i; j++) for (let c = 0; c < DH; c++) cat.data[i * D + h * DH + c] += a[j] * v.data[j * D + h * DH + c];
    }
    heads.push(A);

    if (QKV && l === QKV_LAYER && h === QKV_HEAD) {
      const n = Math.min(SHOW, DH);
      const slice = (m, i) => "[" + Array.from(m.data.subarray(i * D + h * DH, i * D + h * DH + n), f).join(" ") + (n < DH ? " ..." : "") + " ]";
      console.log(`\n  --- q, k, v for layer ${l}, head ${h} (this head owns columns ${h * DH}..${(h + 1) * DH - 1} of each ${D}-wide result; showing first ${n} of ${DH}) ---`);
      console.log(`  q = LayerNorm(x) . Wq    k = LayerNorm(x) . Wk    v = LayerNorm(x) . Wv`);
      ids.forEach((id, i) => {
        console.log(`  pos ${i} ${show(id)}`);
        console.log(`      q ${slice(q, i)}`);
        console.log(`      k ${slice(k, i)}`);
        console.log(`      v ${slice(v, i)}`);
      });
      const i = T - 1;
      console.log(`\n  How the last token (${show(ids[i])}) scores each token, using the FULL ${DH}-number vectors:`);
      console.log(`      score = (q_last . k_j) / sqrt(${DH})  ->  softmax  ->  weight`);
      const sc = [];
      for (let j = 0; j <= i; j++) {
        let dot = 0;
        for (let c = 0; c < DH; c++) dot += q.data[i * D + h * DH + c] * k.data[j * D + h * DH + c];
        sc.push(dot / Math.sqrt(DH));
      }
      const wts = softmax(sc);
      sc.forEach((s, j) => console.log(`      vs pos ${j} ${show(ids[j]).padEnd(8)} score ${f(s)}   weight ${(wts[j] * 100).toFixed(1).padStart(5)}%`));
      const mixed = Array.from({ length: n }, (_, c) => wts.reduce((s, w, j) => s + w * v.data[j * D + h * DH + c], 0));
      console.log(`      output = sum(weight_j * v_j) = [${mixed.map(f).join(" ")}${n < DH ? " ..." : ""} ]`);
    }
  }
  trace.push(heads);
  console.log("  Who looks at whom (each head's attention from the LAST token; rows sum to 1):");
  for (let h = 0; h < NH; h++) {
    const a = heads[h][T - 1];
    const bars = a.map((w, j) => `${show(ids[j])}:${(w * 100).toFixed(0)}%`).join("  ");
    console.log(`    head ${h}: ${bars}`);
  }
  const attnOut = matmul(cat, p("attn.wo")); addBias(attnOut, p("attn.bo"));
  x = add(x, attnOut);
  console.log(`  after attention + residual, last token: ${vec(x, T - 1)}`);

  hr(`LAYER ${l}: feed-forward block (x = x + W2 . GELU(W1 . LayerNorm(x)))`);
  const h1 = matmul(layerNorm(x, p("ln2.gain"), p("ln2.shift")), p("mlp.w1")); addBias(h1, p("mlp.b1"));
  const act = mat(h1.rows, h1.cols);
  for (let i = 0; i < h1.data.length; i++) act.data[i] = gelu(h1.data[i]);
  const last = Array.from(act.data.subarray((T - 1) * act.cols, T * act.cols));
  const active = last.filter((v) => v > 0.1).length;
  console.log(`  expand ${D} -> ${h1.cols} numbers; ${active} of ${h1.cols} neurons are clearly "on" (> 0.1) for the last token`);
  const out = matmul(act, p("mlp.w2")); addBias(out, p("mlp.b2"));
  x = add(x, out);
  console.log(`  after feed-forward + residual, last token: ${vec(x, T - 1)}`);
}

hr("FINAL: LayerNorm, then multiply by `head` (d_model x vocab) -> one score per token");
const logits = matmul(layerNorm(x, W["lnf.gain"], W["lnf.shift"]), W.head);
const lastLogits = Array.from(logits.data.subarray((T - 1) * V, T * V));
const probs = softmax(lastLogits);
console.log(`  the last position produced ${V} scores. Softmax turns them into probabilities:\n`);
const order = probs.map((p, id) => [p, id]).sort((a, b) => b[0] - a[0]).slice(0, TOP);
for (const [p, id] of order)
  console.log(`  ${(p * 100).toFixed(1).padStart(5)}%  ${"#".repeat(Math.round(p * 40)).padEnd(40)} id ${String(id).padStart(3)} ${show(id)}`);
console.log(`\nNext token the model would pick (greedy): ${show(order[0][1])}`);
