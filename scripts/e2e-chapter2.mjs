// End-to-end test for the textbook chapter 2 (embeddings), run with scripts/run-e2e.mjs:
//   node scripts/run-e2e.mjs http://localhost:8080/textbook/chapter2.html scripts/e2e-chapter2.mjs
// Needs: .\scripts\build-web.ps1 done and `python -m http.server 8080 --directory web` running.
// Environment: E2E_STEPS (default 1500), E2E_DARK / E2E_MOBILE for visual variants, E2E_TAG for the screenshot suffix.

export default async function run(page) {
  const steps = process.env.E2E_STEPS || "1500";
  const tag = process.env.E2E_TAG || "";
  const result = { checks: {}, timings: {}, info: {} };
  const ok = (name, pass, detail) => (result.checks[name] = pass ? "pass" : `FAIL ${detail ?? ""}`);
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));

  if (process.env.E2E_DARK) await page.emulateMedia({ colorScheme: "dark" });
  if (process.env.E2E_MOBILE) await page.setViewportSize({ width: 390, height: 800 });

  const state = () => page.evaluate(() => JSON.parse(document.body.dataset.book ?? "null"));
  const status = () => page.textContent("#status");
  const setSlider = (v) => page.evaluate((x) => { const k = document.querySelector("#snap"); k.value = x; k.dispatchEvent(new Event("input")); }, v);
  const neighborsText = (sel) => page.$$eval(`${sel} li`, (els) => els.map((e) => e.querySelector(".chip").textContent));

  // ---- before training ----
  await page.waitForFunction(() => document.querySelector("#status").textContent.startsWith("Ready"), null, { timeout: 60000 });
  let s = await state();
  ok("starts at step 0 with one snapshot", s.snapshots === 1 && s.step === 0 && !s.trained, JSON.stringify(s));
  ok("vocabulary is 512 and dim is 32", s.vocab === 512 && s.dim === 32, JSON.stringify(s));
  ok("default token is ' the'", s.selectedText === " the", String(s.selectedText));
  ok("random start: no strong neighbour", s.top < 0.75, String(s.top));
  ok("vector drawn (32 bars)", (await page.locator("#vec rect").count()) === 32);
  ok("map has a dot per token", (await page.locator("#map circle.pt").count()) === 512);
  ok("train button enabled", await page.isEnabled("#btn-train"));
  ok("slider disabled before training", await page.isDisabled("#snap"));
  result.info.model_line = await page.textContent("#model-line");

  // an unknown token gives an explanation, not an error
  await page.fill("#token-input", "zzqx");
  ok("unknown token explains itself", (await page.textContent("#token-note")).startsWith('No token is exactly "zzqx"'), await page.textContent("#token-note"));
  await page.fill("#token-input", "dog"); // finds " dog" by adding the leading space
  s = await state();
  ok("typing 'dog' finds ' dog'", s.selectedText === " dog", String(s.selectedText));
  await page.fill("#token-input", " the");

  // ---- train ----
  await page.selectOption("#steps", steps);
  const t0 = Date.now();
  await page.click("#btn-train");
  ok("controls locked while training", await page.isDisabled("#btn-train"));
  await page.waitForFunction(() => document.querySelector("#status").textContent.startsWith("Done"), null, { timeout: 280000 });
  result.timings.train_s = +((Date.now() - t0) / 1000).toFixed(1);
  result.info.done_status = await status();

  s = await state();
  ok("11 snapshots and the last is selected", s.snapshots === 11 && s.idx === 10 && s.trained, JSON.stringify(s));
  ok("trained the requested steps", s.step === Number(steps), String(s.step));
  ok("loss went down to a sane value", s.loss > 0 && s.loss < 5, String(s.loss));
  ok("trained neighbours are far closer than random", s.top > 0.8, String(s.top));
  result.info.the_now = await neighborsText("#nn-now");
  result.info.the_before = await neighborsText("#nn-before");

  // look at a few tokens after training
  for (const w of [" dog", " was", ".", " happy", " girl"]) {
    await page.fill("#token-input", w);
    result.info[`neighbors of '${w}'`] = await neighborsText("#nn-now");
  }
  await page.fill("#token-input", " dog");
  const dogState = await state();
  const dogNow = dogState.neighbors;
  const dogText = await neighborsText("#nn-now");
  ok("' dog' has an animal or person among its top 8 neighbours", dogText.some((t) => ["·cat", "·bird", "·girl", "·Tom", "·Tim", "·friend", "·mom"].includes(t)), dogText.join(" "));
  await page.fill("#token-input", " was");
  const wasText = await neighborsText("#nn-now");
  ok("' was' has a form of 'to be' among its top 8 neighbours", wasText.some((t) => ["·were", "·is", "·are"].includes(t)), wasText.join(" "));
  await page.fill("#token-input", ".");
  const dotText = await neighborsText("#nn-now");
  ok("'.' has other punctuation among its top 8 neighbours", dotText.some((t) => /[.!?]/.test(t)), dotText.join(" "));
  await page.fill("#token-input", " dog");

  // ---- scrubbing ----
  await setSlider(0);
  const s0 = await state();
  ok("slider back to step 0", s0.idx === 0 && s0.step === 0, JSON.stringify(s0));
  ok("step 0 neighbours differ from trained ones", JSON.stringify(s0.neighbors) !== JSON.stringify(dogNow));
  ok("step 0 'before' and 'now' lists match", JSON.stringify(await neighborsText("#nn-before")) === JSON.stringify(await neighborsText("#nn-now")));
  await setSlider(10);

  // ---- map interaction ----
  const before = (await state()).selected;
  await page.evaluate(() => { const c = [...document.querySelectorAll("#map circle.pt")].find((e) => !e.classList.contains("sel")); c.dispatchEvent(new MouseEvent("click", { bubbles: true })); });
  ok("clicking a dot selects that token", (await state()).selected !== before);
  ok("token box never shows a control character", !/[\x00-\x1f]/.test(await page.inputValue("#token-input")), JSON.stringify(await page.inputValue("#token-input")));
  const coords = await page.$$eval("#map circle.pt", (cs) => cs.map((c) => [Number(c.getAttribute("cx")), Number(c.getAttribute("cy"))]));
  ok("all map coordinates are finite and inside the frame", coords.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y) && x >= 0 && x <= 600 && y >= 0 && y <= 420), "out of range");
  result.info.map_note = await page.textContent("#map-var");

  await page.screenshot({ path: `textbook-ch2${tag}.png`, fullPage: true });

  // ---- train again starts fresh ----
  if (!process.env.E2E_SKIP_RETRAIN) {
    await page.selectOption("#steps", "500");
    await page.click("#btn-train");
    await page.waitForFunction(() => document.querySelector("#status").textContent.startsWith("Done"), null, { timeout: 280000 });
    s = await state();
    ok("train again restarts from step 0 with fresh snapshots", s.snapshots === 11 && s.step === 500, JSON.stringify(s));
    await page.fill("#token-input", " the");
    await setSlider(0);
    ok("same random start after restart", JSON.stringify(await neighborsText("#nn-now")) === JSON.stringify(result.info.the_before), "random start changed");
  }

  ok("no page errors", errors.length === 0, errors.join(" | "));
  result.failures = Object.entries(result.checks).filter(([, v]) => v !== "pass").length;
  return result;
}
