// End-to-end test for the textbook chapter 4 (learning), run with scripts/run-e2e.mjs:
//   node scripts/run-e2e.mjs http://localhost:8080/textbook/chapter4.html scripts/e2e-chapter4.mjs
// Needs: .\scripts\build-web.ps1 done and `python -m http.server 8080 --directory web` running.
// Environment: E2E_STEPS (default 500), E2E_DARK / E2E_MOBILE for visual variants, E2E_TAG for the screenshot suffix.

export default async function run(page) {
  const steps = process.env.E2E_STEPS || "500";
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

  // ---- the toy ----
  let s = await state();
  ok("toy starts at w = -3 with no steps", s.toy.w === -3 && s.toy.steps === 0, JSON.stringify(s.toy));
  await page.click("#toy-step");
  s = await state();
  // w - 0.3 * 2 * (w - 3) with w = -3  ->  0.6
  ok("one gradient step lands on 0.6", near(s.toy.w, 0.6, 1e-9) && s.toy.steps === 1, JSON.stringify(s.toy));
  await page.click("#toy-run");
  s = await state();
  ok("21 steps at rate 0.3 reach w = 3", near(s.toy.w, 3, 0.01) && s.toy.steps === 21, JSON.stringify(s.toy));
  await page.click("#toy-reset");
  s = await state();
  ok("reset returns to the start", s.toy.w === -3 && s.toy.steps === 0);
  await setRange("lr", "0.9");
  await page.click("#toy-run");
  s = await state();
  ok("rate 0.9 overshoots but still arrives (6 x 0.8^20 = 0.07 away)", near(s.toy.w, 3, 0.1) && s.toy.w < 3, JSON.stringify(s.toy));
  await setRange("lr", "1.05");
  await page.click("#toy-reset");
  await page.click("#toy-run");
  s = await state();
  ok("rate 1.05 diverges (ends further from 3 than it began)", Math.abs(s.toy.w - 3) > 6, JSON.stringify(s.toy));
  ok("the toy says so when the ball leaves the chart", (await page.textContent("#toy-line")).includes("off the chart"), await page.textContent("#toy-line"));
  await setRange("lr", "0.3");
  await setRange("w0", "6");
  s = await state();
  ok("moving the start slider resets the ball", s.toy.w === 6 && s.toy.steps === 0, JSON.stringify(s.toy));
  await setRange("w0", "-3");

  // ---- untrained model ----
  await page.waitForFunction(() => document.querySelector("#status").textContent.startsWith("Ready"), null, { timeout: 90000 });
  s = await state();
  const T = s.T;
  ok("a sentence gives at least 10 tokens", T >= 10, String(T));
  ok("untrained loss is near ln 512 = 6.24", s.meanLoss > 5.5 && s.meanLoss < 7, String(s.meanLoss));
  ok("untrained held-out loss is also near 6.24", s.valPts.length === 1 && s.valPts[0][1] > 5.5 && s.valPts[0][1] < 7, JSON.stringify(s.valPts));
  ok("starts on the second-to-last token", s.selected === T - 2, `${s.selected} of ${T}`);
  ok("one shaded button per token", (await page.locator("#surprise button").count()) === T);
  ok("exactly one button is pressed", (await page.locator('#surprise button[aria-pressed="true"]').count()) === 1);
  ok("eight guesses plus maybe the real token", [8, 9].includes(await page.locator("#guess-list li").count()));
  ok("an untrained model's top 8 guesses hold little probability", s.probSum < 0.2, String(s.probSum));
  ok("snapshot slider locked before training", await page.isDisabled("#snap"));
  ok("the model has the advertised size (about 60,000 numbers)", s.params > 50000 && s.params < 70000, String(s.params));
  const untrainedMean = s.meanLoss;

  await page.click("#surprise button:nth-child(5)");
  s = await state();
  ok("clicking a token selects it", s.selected === 4 && (await page.locator('#surprise button[aria-pressed="true"]').count()) === 1, String(s.selected));
  ok("the guess title names the token", (await page.textContent("#guess-title")).startsWith("After"));

  // odd inputs
  await page.fill("#text", "");
  ok("empty text shows a message", (await page.textContent("#mean-line")).includes("a little more text"), await page.textContent("#mean-line"));
  ok("and clears the token row", (await page.locator("#surprise button").count()) === 0);
  await page.fill("#text", "a");
  ok("one token is not enough", (await page.textContent("#mean-line")).includes("a little more text"));
  await page.fill("#text", "日本語 🌍");
  s = await state();
  ok("non-English text still works", s.T > 1 && Number.isFinite(s.meanLoss), JSON.stringify({ T: s.T, m: s.meanLoss }));
  await page.fill("#text", "the dog and the cat saw the ball. ".repeat(30));
  s = await state();
  ok("very long text is cut to the 64-token window", s.T === 64 && !(await page.locator("#trunc-note").isHidden()), String(s.T));
  await page.fill("#text", "Lily and her dog went to the park. She gave the dog a ball.");
  s = await state();
  ok("restoring the sentence restores the loss", near(s.meanLoss, untrainedMean, 1e-4), `${s.meanLoss} vs ${untrainedMean}`);

  // ---- train ----
  await page.selectOption("#steps", steps);
  const t0 = Date.now();
  await page.click("#btn-train");
  ok("train button and menu locked while training", (await page.isDisabled("#btn-train")) && (await page.isDisabled("#steps")));
  await page.waitForFunction(() => document.querySelector("#status").textContent.startsWith("Done"), null, { timeout: 280000 });
  result.timings.train_s = +((Date.now() - t0) / 1000).toFixed(1);
  s = await state();
  const last = s.valPts[s.valPts.length - 1];
  ok("training finished", s.trained && s.total === Number(steps), JSON.stringify({ t: s.trained, total: s.total }));
  ok("held-out loss fell by more than 1.5", s.valPts[0][1] - last[1] > 1.5, JSON.stringify([s.valPts[0], last]));
  ok("held-out loss ends in the right range (below 4.5)", last[1] < 4.5, String(last[1]));
  ok("held-out checkpoints were recorded", s.valPts.length >= 8 && last[0] === Number(steps), String(s.valPts.length));
  ok("held-out loss never rose by more than 0.3 between checkpoints", s.valPts.every((p, i) => i === 0 || p[1] - s.valPts[i - 1][1] < 0.3), JSON.stringify(s.valPts.map((p) => p[1])));
  ok("a training-loss point was kept for every pair of steps", s.trainPts >= Number(steps) / 2 - 1, String(s.trainPts));
  ok("snapshots start at 0 and end at the last step", s.snaps[0] === 0 && s.snaps[s.snaps.length - 1] === Number(steps) && s.snaps.length >= 4, JSON.stringify(s.snaps));
  ok("the slider jumps to the final snapshot", s.snapIdx === s.snaps.length - 1 && !(await page.isDisabled("#snap")));
  ok("trained text loss is more than 1.5 below untrained", s.meanLoss < untrainedMean - 1.5, `${s.meanLoss} vs ${untrainedMean}`);
  ok("trained top 8 guesses hold much more probability", s.probSum > 0.3, String(s.probSum));
  ok("the chart has one point per held-out checkpoint", (await page.locator("#curve circle.valpt").count()) === s.valPts.length);
  ok("the chart draws both lines and the snapshot marker", (await page.locator("#curve polyline.train").count()) === 1 && (await page.locator("#curve polyline.val").count()) === 1 && (await page.locator("#curve line.mark").count()) === 1);
  await page.screenshot({ path: `textbook-ch4${tag}.png`, fullPage: true });

  // scrub back to step 0: must be exactly the untrained model again
  await setRange("snap", "0");
  s = await state();
  ok("slider at 0 gives the untrained loss exactly", s.snapIdx === 0 && near(s.meanLoss, untrainedMean, 1e-4), `${s.meanLoss} vs ${untrainedMean}`);
  ok("slider label shows the step", (await page.textContent("#snap-out")) === "0");
  await setRange("snap", "2");
  const mid = await state();
  ok("an in-between snapshot is better than untrained", mid.meanLoss < untrainedMean && mid.snapIdx === 2, JSON.stringify({ m: mid.meanLoss, u: untrainedMean }));
  await setRange("snap", String(s.snaps.length - 1));
  s = await state();
  ok("and forward again to the trained model", s.meanLoss < untrainedMean - 1.5);

  ok("no page errors", errors.length === 0, errors.join(" | "));
  result.failures = Object.entries(result.checks).filter(([, v]) => v !== "pass").length;
  return result;
}
