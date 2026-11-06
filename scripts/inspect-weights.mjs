// Print every weight tensor in a checkpoint: name, shape, parameter count,
// simple statistics, and the first few raw values.
//
// Usage:  node scripts/inspect-weights.mjs [run-dir-or-model.safetensors] [--values N]
// Example: node scripts/inspect-weights.mjs runs/demo

import { readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const args = process.argv.slice(2);
const vIdx = args.indexOf("--values");
const nValues = vIdx >= 0 ? Number(args.splice(vIdx, 2)[1]) || 6 : 6;
let target = args[0] ?? "runs/demo";
if (statSync(target).isDirectory()) target = join(target, "model.safetensors");

// safetensors layout: 8-byte little-endian header length, JSON header, raw data.
const buf = readFileSync(target);
const headerLen = Number(buf.readBigUInt64LE(0));
const header = JSON.parse(buf.subarray(8, 8 + headerLen).toString("utf8"));
const body = buf.subarray(8 + headerLen);

const meta = header.__metadata__ ?? {};
delete header.__metadata__;

console.log(`File: ${target} (${(buf.length / 1024).toFixed(1)} KiB)`);
console.log("Metadata:", meta, "\n");

const f = (x) => (x >= 0 ? " " : "") + x.toFixed(4);
const rows = [];
let total = 0;

for (const [name, t] of Object.entries(header)) {
  if (t.dtype !== "F32") throw new Error(`${name}: expected F32, got ${t.dtype}`);
  const [start, end] = t.data_offsets;
  // Buffer.slice is a view, so copy the bytes to get an aligned Float32Array.
  const data = new Float32Array(Uint8Array.prototype.slice.call(body, start, end).buffer);
  let sum = 0, sumSq = 0, min = Infinity, max = -Infinity;
  for (const v of data) {
    sum += v; sumSq += v * v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const mean = sum / data.length;
  const std = Math.sqrt(Math.max(0, sumSq / data.length - mean * mean));
  total += data.length;
  rows.push({ name, shape: t.shape.join(" x "), count: data.length, mean, std, min, max, first: Array.from(data.slice(0, nValues)) });
}

const w = Math.max(...rows.map((r) => r.name.length));
for (const r of rows) {
  console.log(`${r.name.padEnd(w)}  shape ${r.shape.padEnd(10)} params ${String(r.count).padStart(8)}`);
  console.log(`${" ".repeat(w)}  mean ${f(r.mean)}  std ${f(r.std)}  min ${f(r.min)}  max ${f(r.max)}`);
  console.log(`${" ".repeat(w)}  first values: ${r.first.map(f).join(" ")}`);
}

console.log(`\nTotal parameters: ${total.toLocaleString()}  (${rows.length} tensors, ${(total * 4 / 1024).toFixed(1)} KiB as f32)`);
