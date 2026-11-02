// End-to-end test for the textbook chapter 5 (sampling), run with scripts/run-e2e.mjs:
//   node scripts/run-e2e.mjs http://localhost:8080/textbook/chapter5.html scripts/e2e-chapter5.mjs
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
  const setRange = (id, v) => page.evaluate(([id, v]) => { const e = document.getElementById(id); e.value = v; e.dispatchEvent(new Event("input", { bubbles: true })); }, [id, v]);
  const near = (a, b, tol) => Math.abs(a - b) <= tol;

  // ---- untrained model ----
  await page.waitForFunction(() => document.querySelector("#status").textContent.startsWith("Ready"), null, { timeout: 90000 });
  let s = await state();
  ok("untrained: chances sum to 1 over all 512 tokens", near(s.probSum, 1, 1e-3) && s.nonzero === 512, JSON.stringify({ sum: s.probSum, nz: s.nonzero }));
  ok("untrained: nearly flat (uncertainty above 8 bits, favourite under 5%)", s.entropy > 8 && s.top[0] < 0.05, JSON.stringify({ H: s.entropy, top: s.top[0] }));
  ok("write-again button locked before training", await page.isDisabled("#btn-gen"));
  ok("twelve bars are drawn", (await page.locator("#dist li").count()) === 12);
  ok("bars are in falling order", s.top.every((p, i) => i === 0 || s.top[i - 1] >= p), JSON.stringify(s.top));
  ok("temperature and top-k labels start at 1.00 and all", (await page.textContent("#temp-out")) === "1.00" && (await page.textContent("#topk-out")) === "all");
  const untrainedH = s.entropy;

  // ---- train ----
  await page.selectOption("#steps", steps);
  const t0 = Date.now();
  await page.click("#btn-train");
  ok("train button locked while training", await page.isDisabled("#btn-train"));
  await page.waitForFunction(() => document.querySelector("#status").textContent.startsWith("Done"), null, { timeout: 280000 });
  result.timings.train_s = +((Date.now() - t0) / 1000).toFixed(1);
  const base = await state();
  ok("trained: still a proper distribution", near(base.probSum, 1, 1e-3) && base.nonzero === 512);
  ok("trained: more certain than untrained (3+ bits less uncertainty)", base.entropy < untrainedH - 3, `${base.entropy} vs ${untrainedH}`);
  ok("trained: a clear favourite (over 15%)", base.top[0] > 0.15, String(base.top[0]));
  ok("write-again button unlocked", !(await page.isDisabled("#btn-gen")));
  ok("three generations were written, all non-empty", base.gens.length === 3 && base.gens.every((g) => g.length > 20), JSON.stringify(base.gens.map((g) => g.length)));
  ok("cool and hot text differ", base.gens[0] !== base.gens[2]);
  ok("a seed is shown", /seed \d+/.test(await page.textContent("#seed-out")));
  await page.screenshot({ path: `textbook-ch5${tag}.png`, fullPage: true });

  // ---- temperature ----
  await setRange("temp", "0.1");
  const cold = await state();
  await setRange("temp", "3");
  const hot = await state();
  ok("label follows the slider", (await page.textContent("#temp-out")) === "3.00");
  ok("every temperature is still a proper distribution", [cold, hot].every((d) => near(d.probSum, 1, 1e-3) && d.nonzero === 512), JSON.stringify([cold.probSum, hot.probSum]));
  ok("low temperature sharpens: bigger favourite, less uncertainty", cold.top[0] > base.top[0] && cold.entropy < base.entropy, JSON.stringify({ c: cold.top[0], b: base.top[0] }));
  ok("high temperature flattens: smaller favourite, more uncertainty", hot.top[0] < base.top[0] && hot.entropy > base.entropy, JSON.stringify({ h: hot.top[0], b: base.top[0] }));
  ok("temperature never changes which token is the favourite", cold.favourite === base.favourite && hot.favourite === base.favourite);
  await setRange("temp", "1");
  s = await state();
  ok("back at 1.00 gives the original chances", JSON.stringify(s.top) === JSON.stringify(base.top));

  // ---- top-k ----
  await setRange("topk", "5");
  const k5 = await state();
  ok("top-k 5 leaves exactly 5 possible tokens", k5.k === 5 && k5.nonzero === 5 && near(k5.probSum, 1, 1e-3), JSON.stringify({ k: k5.k, nz: k5.nonzero }));
  ok("top-k label shows 5", (await page.textContent("#topk-out")) === "5");
  ok("top-k raises the favourite's share", k5.top[0] > base.top[0]);
  ok("bars beyond k show ~0", (await page.locator("#dist li .num", { hasText: "~0" }).count()) === 7);

  // ---- rolling ----
  ok("no ticks before rolling", (await page.locator("#dist .tick-obs").count()) === 0);
  await page.click("#btn-many");
  s = await state();
  const p = s.top[0];
  const sigma = Math.sqrt(1000 * p * (1 - p));
  ok("1,000 rolls were counted", s.counts === 1000, String(s.counts));
  ok("favourite came up about as often as its chance says (within 5 sigma)", near(s.favouriteCount, 1000 * p, 5 * sigma + 1), `${s.favouriteCount} vs ${(1000 * p).toFixed(0)}`);
  ok("a tick sits on every bar", (await page.locator("#dist .tick-obs").count()) === 12);
  await page.click("#btn-one");
  s = await state();
  ok("one roll picks a token", Number.isInteger(s.lastRoll), String(s.lastRoll));
  ok("the roll line names the token", (await page.textContent("#roll-line")).startsWith("The die picked"));
  await setRange("topk", "1");
  s = await state();
  ok("top-k 1 is greedy: one token with probability 1", s.nonzero === 1 && s.top[0] === 1, JSON.stringify({ nz: s.nonzero, top: s.top[0] }));
  ok("moving a dial clears old rolls", (await page.locator("#dist .tick-obs").count()) === 0 && (await page.textContent("#roll-line")) === "");
  await page.click("#btn-one");
  s = await state();
  ok("greedy always picks the favourite", s.lastRoll === s.favourite, JSON.stringify({ r: s.lastRoll, f: s.favourite }));
  await setRange("topk", "41");

  // ---- prompts and odd inputs ----
  await page.fill("#text", "");
  ok("empty prompt shows a message", (await page.textContent("#dist-line")).includes("type some text") && (await page.locator("#dist li").count()) === 0, await page.textContent("#dist-line"));
  await page.fill("#text", "日本語 🌍");
  s = await state();
  ok("non-English prompt still gives a distribution", near(s.probSum, 1, 1e-3));
  await page.fill("#text", "the dog and the cat saw the ball. ".repeat(30));
  s = await state();
  ok("very long prompt is cut and says so", near(s.probSum, 1, 1e-3) && !(await page.locator("#trunc-note").isHidden()));
  await page.fill("#text", "Once upon a time, there was a little");
  s = await state();
  ok("restoring the prompt restores the chances", JSON.stringify(s.top) === JSON.stringify(base.top));

  // ---- write again ----
  const before = (await state()).gens;
  const seedBefore = await page.textContent("#seed-out");
  await page.click("#btn-gen");
  s = await state();
  ok("write again uses a new seed and new text", (await page.textContent("#seed-out")) !== seedBefore && JSON.stringify(s.gens) !== JSON.stringify(before));
  await page.fill("#text", "");
  await page.click("#btn-gen");
  s = await state();
  ok("write again with an empty prompt shows a message, not a crash", s.gens.every((g) => typeof g === "string" && g.length > 0));
  await page.fill("#text", "Once upon a time, there was a little");

  ok("no page errors", errors.length === 0, errors.join(" | "));
  result.failures = Object.entries(result.checks).filter(([, v]) => v !== "pass").length;
  return result;
}
