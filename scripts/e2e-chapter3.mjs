// End-to-end test for the textbook chapter 3 (attention), run with scripts/run-e2e.mjs:
//   node scripts/run-e2e.mjs http://localhost:8080/textbook/chapter3.html scripts/e2e-chapter3.mjs
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
  const headTable = () => page.$$eval("#head-rows tr", (rows) => rows.map((r) => [...r.children].map((c) => c.textContent)));
  const pct = (x) => Number(x.replace("%", "")) / 100;

  // ---- untrained model ----
  await page.waitForFunction(() => document.querySelector("#status").textContent.startsWith("Ready"), null, { timeout: 90000 });
  let s = await state();
  const T = s.T;
  ok("a sentence gives at least 10 tokens", T >= 10, String(T));
  ok("starts on the untrained model, last token selected", s.which === "untrained" && !s.trained && s.query === T - 1, JSON.stringify(s));
  ok("selected row's weights add up to 1", Math.abs(s.rowSum - 1) < 1e-4, String(s.rowSum));
  ok("layer and head menus have 2 entries each", (await page.locator("#layer option").count()) === 2 && (await page.locator("#head option").count()) === 2);
  ok("heat map has T(T+1)/2 weight cells", (await page.locator("#heat .cellw").count()) === (T * (T + 1)) / 2, String(await page.locator("#heat .cellw").count()));
  ok("heat map blanks the future: T(T-1)/2 masked cells", (await page.locator("#heat .masked").count()) === (T * (T - 1)) / 2);
  ok("trained option disabled before training", await page.isDisabled('input[name="which"][value="trained"]'));
  const untrainedHeads = await headTable();
  result.info.untrained_heads = untrainedHeads;
  ok("head table has one row per head", untrainedHeads.length === 4);
  ok("untrained heads show no strong previous-token habit", s.summary.every((h) => h.prev < 0.3), JSON.stringify(s.summary.map((h) => h.prev)));

  // row table matches the selected row
  ok("score table has query+1 rows", (await page.locator("#score-rows tr").count()) === s.query + 1);
  ok("weights column totals 1.000", (await page.textContent("#w-total")) === "1.000", await page.textContent("#w-total"));

  // click a row in the heat map
  await page.evaluate(() => document.querySelector('#heat g.row[data-row="5"] .hit').dispatchEvent(new MouseEvent("click", { bubbles: true })));
  s = await state();
  ok("clicking a row selects it", s.query === 5, String(s.query));
  ok("score table follows the click", (await page.locator("#score-rows tr").count()) === 6);
  ok("row title names the token and layer", /layer 1, head 1/.test(await page.textContent("#row-title")), await page.textContent("#row-title"));

  // layer and head menus
  await page.selectOption("#layer", "1");
  await page.selectOption("#head", "1");
  s = await state();
  ok("menus change layer and head", s.layer === 1 && s.head === 1 && Math.abs(s.rowSum - 1) < 1e-4, JSON.stringify(s));
  ok("current head is highlighted in the table", (await page.locator("#head-rows tr.current").count()) === 1);
  await page.selectOption("#layer", "0");
  await page.selectOption("#head", "0");

  // odd inputs
  await page.fill("#text", "");
  ok("empty text shows a message instead of breaking", (await page.textContent("#row-title")).includes("type some text"), await page.textContent("#row-title"));
  await page.fill("#text", "日本語 🌍");
  s = await state();
  ok("non-English text still gives a valid table", s.T > 0 && Math.abs(s.rowSum - 1) < 1e-4, JSON.stringify(s));
  await page.fill("#text", "the dog and the cat saw the ball. ".repeat(30));
  s = await state();
  ok("very long text is cut to the 64-token window", s.T === 64 && s.truncated && !(await page.locator("#trunc-note").isHidden()), JSON.stringify({ T: s.T, tr: s.truncated }));
  await page.fill("#text", "Lily and her dog went to the park. She gave the dog a ball.");

  // ---- train ----
  await page.selectOption("#steps", steps);
  const t0 = Date.now();
  await page.click("#btn-train");
  ok("train button locked while training", await page.isDisabled("#btn-train"));
  await page.waitForFunction(() => document.querySelector("#status").textContent.startsWith("Done"), null, { timeout: 280000 });
  result.timings.train_s = +((Date.now() - t0) / 1000).toFixed(1);
  s = await state();
  ok("switches to the trained model", s.trained && s.which === "trained", JSON.stringify(s));
  const trainedHeads = await headTable();
  result.info.trained_heads = trainedHeads;
  ok("training changed the head table", JSON.stringify(trainedHeads) !== JSON.stringify(untrainedHeads));
  ok("some trained head clearly prefers the previous token (>= 15%)", s.summary.some((h) => h.prev >= 0.15), JSON.stringify(s.summary.map((h) => +h.prev.toFixed(2))));
  ok("weights still add up to 1 after training", Math.abs(s.rowSum - 1) < 1e-4, String(s.rowSum));
  result.info.even_note = await page.textContent("#even-note");
  await page.screenshot({ path: `textbook-ch3${tag}.png`, fullPage: true });

  // compare back to untrained
  await page.check('input[name="which"][value="untrained"]');
  s = await state();
  ok("can switch back to the untrained model", s.which === "untrained" && JSON.stringify(await headTable()) === JSON.stringify(untrainedHeads));
  await page.check('input[name="which"][value="trained"]');
  ok("and forward again", JSON.stringify(await headTable()) === JSON.stringify(trainedHeads));

  ok("no page errors", errors.length === 0, errors.join(" | "));
  result.failures = Object.entries(result.checks).filter(([, v]) => v !== "pass").length;
  return result;
}
