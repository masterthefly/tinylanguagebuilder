// Runs one end-to-end script against the web app and prints its result as JSON.
//
//   cd scripts && npm install          (once: installs Playwright)
//   npx playwright install chromium    (once: downloads the browser)
//   node scripts/run-e2e.mjs http://localhost:8080/ scripts/e2e-web.mjs
//   node scripts/run-e2e.mjs http://localhost:8080/textbook/chapter3.html scripts/e2e-chapter3.mjs
//
// Before running: build the web app (.\scripts\build-web.ps1) and serve it
// (python -m http.server 8080 --directory web).
//
// Each e2e script exports `default async function run(page)`, where `page` is a Playwright Page that has
// already loaded the URL, and returns an object with a `checks` map and a `failures` count. This runner exits
// with status 1 if any check failed or the script threw, so it can be used in CI.
//
// Options:   --timeout <ms>   page load timeout (default 60000)     --headed   show the browser window
// Environment: E2E_BROWSER_PATH  use this Chrome/Edge/Chromium executable instead of Playwright's own
//              E2E_PLAYWRIGHT    import this package or file instead of "playwright" (any Playwright-compatible one)
//              plus whatever the script itself reads (E2E_STEPS, E2E_MOBILE, E2E_DARK, E2E_TAG, ...).

import path from "node:path";
import { pathToFileURL } from "node:url";

const args = process.argv.slice(2);
const flag = (name) => {
  const i = args.indexOf(name);
  if (i < 0) return null;
  const [, value] = args.splice(i, 2);
  return value;
};
const headed = args.includes("--headed");
if (headed) args.splice(args.indexOf("--headed"), 1);
const timeout = Number(flag("--timeout") ?? 60000);
const [url, scriptPath] = args;
if (!url || !scriptPath) {
  console.error("usage: node scripts/run-e2e.mjs <url> <script.mjs> [--timeout ms] [--headed]");
  process.exit(2);
}

const { chromium } = await import(process.env.E2E_PLAYWRIGHT ?? "playwright");
const script = await import(pathToFileURL(path.resolve(scriptPath)).href);

const browser = await chromium.launch({ headless: !headed, executablePath: process.env.E2E_BROWSER_PATH || undefined });
let result;
try {
  const context = await browser.newContext({ acceptDownloads: true, viewport: { width: 1280, height: 720 } });
  const page = await context.newPage();
  await page.goto(url, { waitUntil: "domcontentloaded", timeout });
  result = await script.default(page);
} catch (e) {
  result = { error: String(e?.stack ?? e) };
} finally {
  await browser.close();
}

console.log(JSON.stringify(result, null, 2));
const failed = result?.error || (result?.failures ?? 0) > 0;
console.error(failed ? "\nFAILED" : `\nPassed ${Object.keys(result?.checks ?? {}).length} checks.`);
process.exit(failed ? 1 : 0);
