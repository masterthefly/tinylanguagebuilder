// Time training speed of a built wasm package, in Node (same V8 WebAssembly engine as Chrome).
//   node scripts/bench-wasm.mjs web/pkg
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { join, resolve } from "node:path";

const dir = resolve(process.argv[2] || "web/pkg");
const glue = await import(pathToFileURL(join(dir, "tlm_wasm.js")).href);
glue.initSync({ module: readFileSync(join(dir, "tlm_wasm_bg.wasm")) });

const text = readFileSync("web/data/tinystories-8mb.txt", "utf8");
const session = new glue.Session(text, JSON.stringify({ steps: 100000 }));
session.step(20); // warm up

let best = Infinity;
for (let round = 0; round < 5; round++) {
  const t = performance.now();
  session.step(40);
  best = Math.min(best, (performance.now() - t) / 40);
}
console.log(`${dir}: ${best.toFixed(2)} ms/step  ->  ${(1000 / best).toFixed(0)} steps/s  (best of 5 rounds)`);
