// Makes the sample text the web app and the textbook load: the first 8 MB of TinyStories, cut after the
// last complete story, so the browser has a small, clean file to train on. Works on any OS.
//
//   node scripts/make-sample.mjs [input] [output]
//
// Defaults: input  data/tinystories/valid.txt          (see the README for the download)
//           output web/data/tinystories-8mb.txt        (git-ignored; the Pages build recreates it)

import fs from "node:fs";
import path from "node:path";

const [input = "data/tinystories/valid.txt", output = "web/data/tinystories-8mb.txt"] = process.argv.slice(2);
const LIMIT = 8_000_000;
const STOP = "<|endoftext|>";

if (!fs.existsSync(input)) {
  console.error(`Input not found: ${input}\nDownload it first (see "Get training data" in the README).`);
  process.exit(1);
}
const head = fs.readFileSync(input).subarray(0, LIMIT);
const text = new TextDecoder("utf-8").decode(head);      // a character cut by the limit becomes U+FFFD, and is dropped below
const cut = text.lastIndexOf(STOP);
if (cut < 0) {
  console.error(`No "${STOP}" story separator found in the first ${LIMIT} bytes of ${input}.`);
  process.exit(1);
}
const out = text.slice(0, cut + STOP.length) + "\n";
fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, out, "utf8");
console.log(`Wrote ${output}: ${Buffer.byteLength(out).toLocaleString()} bytes, ${out.split(STOP).length - 1} stories.`);
