// End-to-end test for fine-tuning on the user's OWN examples in the browser app, run with scripts/run-e2e.mjs:
//   node scripts/run-e2e.mjs http://localhost:8080/ scripts/e2e-custom.mjs
// Needs: .\scripts\build-web.ps1 done and `python -m http.server 8080 --directory web` running.
// Environment: E2E_STEPS (pre-training steps, default 150), E2E_FT_STEPS (default 150), E2E_DARK / E2E_MOBILE / E2E_TAG.

export default async function run(page) {
  const steps = process.env.E2E_STEPS || "150";
  const ftSteps = process.env.E2E_FT_STEPS || "150";
  const tag = process.env.E2E_TAG || "";
  const result = { checks: {}, timings: {}, info: {} };
  const ok = (name, pass, detail) => (result.checks[name] = pass ? "pass" : `FAIL ${detail ?? ""}`);
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));

  if (process.env.E2E_DARK) await page.emulateMedia({ colorScheme: "dark" });
  if (process.env.E2E_MOBILE) await page.setViewportSize({ width: 390, height: 800 });

  const state = () => page.evaluate(() => JSON.parse(document.body.dataset.tlb));
  const status = () => page.textContent("#status");
  const waitStatus = (needle, timeout = 280000) =>
    page.waitForFunction((n) => document.querySelector("#status").textContent.includes(n), needle, { timeout });
  const file = (name, text) => ({ name, mimeType: "text/plain", buffer: Buffer.from(text, "utf8") });
  const lines = (n) => Array.from({ length: n }, (_, i) => JSON.stringify({ prompt: `Opposite of word${i}:`, response: ` anti${i}` })).join("\n") + "\n";
  const pick = async (text, name = "examples.jsonl") => {
    await page.setInputFiles("#ft-file", file(name, text));
    await page.waitForTimeout(150);
  };

  // ---- prepare and pre-train a small model (the built-in task is the default) ----
  await page.fill("#steps", steps);
  await page.selectOption("#preset", "tiny");
  await page.fill("#steps", steps);
  await page.selectOption("#workers", "2");
  await page.click("#btn-load");
  await page.waitForFunction(() => !document.querySelector("#btn-train").disabled, null, { timeout: 120000 });
  ok("built-in task is selected by default", await page.isChecked('input[name="ft-source"][value="word"]'));
  ok("the file chooser is hidden until asked for", await page.locator("#ft-custom").isHidden());
  let t0 = Date.now();
  await page.click("#btn-train");
  await waitStatus("Finished");
  result.timings.pretrain_s = +((Date.now() - t0) / 1000).toFixed(1);

  // ---- choosing a file ----
  await page.check('input[name="ft-source"][value="file"]');
  ok("choosing 'my own examples' shows the file chooser", await page.locator("#ft-custom").isVisible());
  ok("the format help is available", (await page.locator("#ft-custom details.format").count()) === 1);

  await page.click("#btn-ft");
  ok("starting without a file explains what to do", (await status()).includes("Choose your examples file first"), await status());
  ok("and nothing started", (await state()).phase === "pretrain");

  await pick("this is not json\n");
  let s = await state();
  ok("a bad line is reported with its line number", /line 1 is not valid JSON/.test(await page.textContent("#ft-file-note")), await page.textContent("#ft-file-note"));
  ok("and the note is styled as an error", await page.locator("#ft-file-note.error").count() === 1);
  await page.click("#btn-ft");
  ok("starting with a bad file is refused with the same message", /line 1 is not valid JSON/.test(await status()) && (await state()).phase === "pretrain", await status());

  await pick(lines(5) + JSON.stringify({ prompt: "only a prompt" }) + "\n");
  ok("a line without a response is reported", /line 6 has no "response"/.test(await page.textContent("#ft-file-note")), await page.textContent("#ft-file-note"));

  await pick(lines(5));
  ok("too few examples is reported", /5 examples\. At least 20/.test(await page.textContent("#ft-file-note")), await page.textContent("#ft-file-note"));
  await page.click("#btn-ft");
  ok("and refused", /at least 20/.test(await status()) && (await state()).phase === "pretrain", await status());

  await page.setInputFiles("#ft-file", []);
  await page.waitForTimeout(150);
  ok("clearing the file resets the note", (await page.textContent("#ft-file-note")).includes("Choose a file"));

  // The sample file the page offers must itself be valid.
  await page.click("#ft-custom details.format summary");
  ok("the sample button is visible once the help is opened", await page.locator("#btn-sample-jsonl").isVisible());
  const [download] = await Promise.all([page.waitForEvent("download"), page.click("#btn-sample-jsonl")]);
  const sample = await (await import("node:fs/promises")).readFile(await download.path(), "utf8");
  ok("the downloadable sample is a valid 40-example file", sample.trim().split("\n").length === 40 && sample.trim().split("\n").every((l) => { const v = JSON.parse(l); return typeof v.prompt === "string" && typeof v.response === "string"; }));
  await pick(sample, "opposites-sample.jsonl");
  s = await state();
  ok("a good file is accepted", s.customFile.examples === 40 && s.customFile.error === null, JSON.stringify(s.customFile));
  ok("the note says how many examples", (await page.textContent("#ft-file-note")).includes("40 examples found"));

  // ---- fine-tune on it ----
  await page.fill("#ft-steps", ftSteps);
  t0 = Date.now();
  await page.click("#btn-ft");
  await page.waitForFunction(() => JSON.parse(document.body.dataset.tlb).phase === "finetune", null, { timeout: 60000 });
  s = await state();
  ok("fine-tuning started on the custom task (template is the bare prompt)", s.custom && s.template === "{}", JSON.stringify({ c: s.custom, t: s.template }));
  ok("the source choice is locked while training", await page.isDisabled('input[name="ft-source"][value="word"]'));
  ok("the note says 36 train / 4 held out", /36 training examples, 4 held out/.test(await page.textContent("#ft-note")), await page.textContent("#ft-note"));
  ok("the results table uses the custom wording", (await page.textContent("#ft-row-word")) === "Matches your answer exactly" && await page.locator("#ft-row-chance").isHidden());
  ok("the sample prompt is one of the user's own", (await page.inputValue("#ask-word")).startsWith("Opposite of"), await page.inputValue("#ask-word"));
  ok("before-training examples are shown side by side", (await page.locator("#ft-compare-list li").count()) === 4 && (await page.textContent("#ft-compare-title")).includes("before"));
  await waitStatus("Fine-tuning finished");
  result.timings.finetune_s = +((Date.now() - t0) / 1000).toFixed(1);
  const doneText = await status();
  ok("the finish message talks about matching your answers", /matches the answer you gave exactly/.test(doneText), doneText);
  s = await state();
  ok("fine-tuning loss was recorded and fell", s.ft >= 2 && s.lastFt.val < s.lastFt.train + 5 && Number.isFinite(s.lastFt.val), JSON.stringify(s.lastFt));
  ok("the table has before and after numbers", /%$/.test(await page.textContent("#ft-before-word")) && /%$/.test(await page.textContent("#ft-after-word")));
  ok("after-training examples replace the before ones", (await page.textContent("#ft-compare-title")).includes("after") && (await page.locator("#ft-compare-list li").count()) === 4);
  const items = await page.$$eval("#ft-compare-list li", (els) => els.map((e) => e.textContent));
  ok("each example shows prompt, your answer and the model's", items.every((t) => t.includes("Prompt") && t.includes("Your answer") && t.includes("Model's answer") && t.includes("Opposite of")), items[0]);
  // A custom fine-tune on a few dozen examples overfits; the page must say so exactly when the held-out loss has risen.
  const best = s.ftMinVal;
  const rose = best && s.lastFt.val > best.val * 1.1;
  ok("the memorising warning appears exactly when held-out loss rose 10% above its best", doneText.includes("memorising your examples") === !!rose, JSON.stringify({ best, last: s.lastFt.val, doneText: doneText.slice(-160) }));
  const overflow = await page.evaluate(() => { const p = document.querySelector(".panel.finetune"); return p.scrollWidth - p.clientWidth; });
  ok("the fine-tuning panel does not overflow sideways", overflow <= 1, String(overflow));
  await page.screenshot({ path: `e2e-custom${tag}.png`, fullPage: true });

  // ---- preference tuning is not offered for custom examples, and says why ----
  ok("preference tuning is disabled", await page.isDisabled("#btn-dpo"));
  ok("and the page says why", (await page.textContent("#dpo-hint")).includes("your own examples have none"), await page.textContent("#dpo-hint"));

  // ---- ask it ----
  ok("the ask box is relabelled", (await page.textContent("#ask-label")) === "Your prompt" && /written the way your examples/.test(await page.textContent("#ask-hint")));
  await page.fill("#ask-word", "Opposite of hot:");
  await page.click("#btn-ask");
  await page.waitForFunction(() => document.querySelectorAll("#answers li:not(.empty)").length === 3, null, { timeout: 60000 });
  ok("three answers come back", (await page.locator("#answers li").count()) === 3);

  // ---- downloads carry the task ----
  const [d2] = await Promise.all([page.waitForEvent("download"), page.click("#btn-dl-sft")]);
  const sft = JSON.parse(await (await import("node:fs/promises")).readFile(await d2.path(), "utf8"));
  ok("sft.json records a custom task with the bare template", sft.task === "jsonl" && sft.template === "{}" && sft.word_rate_after === null, JSON.stringify(sft));

  // ---- save, reload, resume ----
  await page.waitForFunction(() => document.querySelector("#save-status").textContent.startsWith("Saved"), null, { timeout: 60000 });
  const before = await page.textContent("#ft-after-word");
  await page.reload();
  await page.waitForSelector("#resume-banner:not([hidden])", { timeout: 20000 });
  await page.click("#btn-resume");
  await waitStatus("Resumed at step", 120000);
  s = await state();
  ok("resume keeps the custom task", s.custom && s.phase === "finetune" && s.template === "{}", JSON.stringify({ c: s.custom, p: s.phase }));
  ok("resume restores the results table", (await page.textContent("#ft-after-word")) === before && (await page.locator("#ft-compare-list li").count()) === 4);
  ok("and the preference tuning lock", await page.isDisabled("#btn-dpo"));
  await page.fill("#ask-word", "Opposite of up:");
  await page.click("#btn-ask");
  await page.waitForFunction(() => document.querySelectorAll("#answers li:not(.empty)").length === 3, null, { timeout: 60000 });
  ok("a resumed custom model still answers", true);

  // ---- a new model goes back to the built-in task ----
  await page.fill("#steps", steps);
  await page.click("#btn-load");
  await page.waitForFunction(() => !document.querySelector("#btn-train").disabled, null, { timeout: 120000 });
  s = await state();
  ok("preparing a new model clears the custom task", !s.custom && s.template === null && !(await page.isDisabled('input[name="ft-source"][value="word"]')), JSON.stringify({ c: s.custom, t: s.template }));
  ok("and the table shows the built-in wording again", (await page.textContent("#ft-row-word")) === "Uses the requested word" && (await page.locator("#ft-compare").isHidden()));

  ok("no page errors", errors.length === 0, errors.join(" | "));
  result.failures = Object.entries(result.checks).filter(([, v]) => v !== "pass").length;
  return result;
}
