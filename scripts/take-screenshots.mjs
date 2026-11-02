// Takes the README screenshots from a running copy of the site.
//
//   node scripts/take-screenshots.mjs [baseUrl] [outDir]
//
// Defaults: http://localhost:8080/  and  docs/screenshots. Needs Playwright (see scripts/run-e2e.mjs) and the site
// served first:  python -m http.server 8080 --directory web
// Environment: E2E_BROWSER_PATH / E2E_PLAYWRIGHT work as in run-e2e.mjs.

import fs from "node:fs";
import path from "node:path";

const base = (process.argv[2] ?? "http://localhost:8080/").replace(/\/?$/, "/");
const out = process.argv[3] ?? "docs/screenshots";
fs.mkdirSync(out, { recursive: true });

const { chromium } = await import(process.env.E2E_PLAYWRIGHT ?? "playwright");
const browser = await chromium.launch({ executablePath: process.env.E2E_BROWSER_PATH || undefined });
const context = await browser.newContext({ viewport: { width: 1200, height: 800 }, deviceScaleFactor: 1.5, colorScheme: "light" });
const shot = (page, name, opts = {}) => page.screenshot({ path: path.join(out, name), ...opts });
const trained = (page, min) => page.waitForFunction(() => document.querySelector("#status").textContent.startsWith("Done"), null, { timeout: 280000 });

// 1. The app: pre-train a small model and show the loss chart and the stories it writes along the way.
{
  const page = await context.newPage();
  await page.goto(base);
  await page.selectOption("#preset", "small");
  await page.fill("#steps", "1200");
  await page.selectOption("#workers", "4");
  await page.click("#btn-load");
  await page.waitForFunction(() => !document.querySelector("#btn-train").disabled, null, { timeout: 120000 });
  await page.click("#btn-train");
  await page.waitForFunction(() => document.querySelector("#status").textContent.includes("Finished"), null, { timeout: 280000 });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.setViewportSize({ width: 1200, height: 1000 });
  await shot(page, "app.png");
  await page.close();
}

// 2. Textbook chapter 3: the attention heat map of a trained model.
{
  const page = await context.newPage();
  await page.goto(base + "textbook/chapter3.html");
  await page.waitForFunction(() => document.querySelector("#status").textContent.startsWith("Ready"), null, { timeout: 90000 });
  await page.selectOption("#steps", "1500");
  await page.click("#btn-train");
  await trained(page);
  await page.evaluate(() => document.querySelector('#heat g.row[data-row="12"] .hit').dispatchEvent(new MouseEvent("click", { bubbles: true })));
  const map = page.locator("section[aria-labelledby=h-map]");
  await map.screenshot({ path: path.join(out, "textbook-attention.png") });
  await page.close();
}

// 3. Textbook chapter 4: the loss curve and the per-token surprise.
{
  const page = await context.newPage();
  await page.goto(base + "textbook/chapter4.html");
  await page.waitForFunction(() => document.querySelector("#status").textContent.startsWith("Ready"), null, { timeout: 90000 });
  await page.selectOption("#steps", "1500");
  await page.click("#btn-train");
  await trained(page);
  const panel = page.locator("section[aria-labelledby=h-train]");
  await panel.screenshot({ path: path.join(out, "textbook-learning.png") });
  await page.close();
}

// 4. The "Explain" popover on a selected term.
{
  const page = await context.newPage();
  await page.setViewportSize({ width: 1200, height: 760 });
  await page.goto(base + "textbook/chapter3.html");
  await page.evaluate(() => {
    const walker = document.createTreeWalker(document.querySelector("main.book"), NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const n = walker.currentNode, i = n.data.indexOf("softmax");
      if (i >= 0) {
        const r = document.createRange();
        r.setStart(n, i);
        r.setEnd(n, i + 7);
        getSelection().removeAllRanges();
        getSelection().addRange(r);
        n.parentElement.scrollIntoView({ block: "center" });
        return;
      }
    }
  });
  await page.waitForFunction(() => !document.getElementById("explain-btn").hidden, null, { timeout: 3000 });
  await page.click("#explain-btn");
  await shot(page, "textbook-explain.png");
  await page.close();
}

await browser.close();
for (const f of fs.readdirSync(out)) console.log(f, Math.round(fs.statSync(path.join(out, f)).size / 1024) + " KB");
