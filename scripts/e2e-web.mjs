// End-to-end test for the web page, run with scripts/run-e2e.mjs:
//   node scripts/run-e2e.mjs http://localhost:8080/ scripts/e2e-web.mjs
// Needs: .\scripts\build-web.ps1 done, `python -m http.server 8080 --directory web` running,
// and (for the CLI compatibility check) `cargo build --release` done.
//
// Environment: E2E_STEPS (default 300), E2E_FT_STEPS (default 300), E2E_PRESET (small),
// E2E_WORKERS (4), E2E_DARK / E2E_MOBILE (visual variants), E2E_TAG (screenshot suffix).

import { execFileSync } from "node:child_process";
import fs from "node:fs";

export default async function run(page) {
  const steps = process.env.E2E_STEPS || "300";
  const ftSteps = process.env.E2E_FT_STEPS || "300";
  const workers = process.env.E2E_WORKERS || "4";
  const tag = process.env.E2E_TAG || "";
  const result = { checks: {}, timings: {} };
  const ok = (name, pass, detail) => (result.checks[name] = pass ? "pass" : `FAIL ${detail ?? ""}`);
  const clock = (name, t0) => (result.timings[name] = +((Date.now() - t0) / 1000).toFixed(1));
  const text = (sel) => page.textContent(sel);
  const status = () => text("#status");
  const waitStatus = (needle, timeout = 280000) =>
    page.waitForFunction((n) => document.querySelector("#status").textContent.includes(n), needle, { timeout });
  const state = () => page.evaluate(() => JSON.parse(document.body.dataset.tlb));
  const prepare = async (nWorkers) => {
    await page.fill("#steps", steps);
    await page.selectOption("#preset", process.env.E2E_PRESET || "small");
    await page.fill("#steps", steps);
    await page.selectOption("#workers", nWorkers);
    await page.click("#btn-load");
    await page.waitForFunction(() => !document.querySelector("#btn-train").disabled, null, { timeout: 120000 });
  };

  if (process.env.E2E_DARK) await page.emulateMedia({ colorScheme: "dark" });
  if (process.env.E2E_MOBILE) await page.setViewportSize({ width: 390, height: 800 });
  fs.mkdirSync("runs/web-export", { recursive: true });

  // ---- 0. a fresh browser has nothing saved ----
  ok("no saved session at first", await page.locator("#resume-banner").isHidden());

  // ---- 1. pre-train with several workers; pause part-way ----
  let t0 = Date.now();
  await prepare(workers);
  clock("prepare", t0);
  result.readyStatus = await status();
  ok("status mentions the workers", result.readyStatus.includes(`${workers} workers`), result.readyStatus);

  t0 = Date.now();
  await page.click("#btn-train");
  await page.waitForFunction((n) => JSON.parse(document.body.dataset.tlb).step >= n, Math.round(Number(steps) * 0.4), { timeout: 200000 });
  await page.click("#btn-pause");
  await page.waitForFunction(() => !document.querySelector("#btn-train").disabled, null, { timeout: 60000 });
  const paused = await state();
  ok("paused part-way", paused.step > 0 && paused.step < Number(steps), JSON.stringify(paused));
  await page.waitForFunction(() => document.querySelector("#save-status").textContent.startsWith("Saved"), null, { timeout: 60000 });

  // ---- 2. reload the page: the session must be offered back, and resume exactly ----
  await page.reload();
  await page.waitForSelector("#resume-banner:not([hidden])", { timeout: 20000 });
  result.banner = await text("#resume-text");
  await page.click("#btn-resume");
  await waitStatus("Resumed at step", 120000);
  const resumed = await state();
  ok("resume restores the step", resumed.step === paused.step, `${resumed.step} vs ${paused.step}`);
  ok("resume restores the chart history", resumed.pre === paused.pre && resumed.pre > 2, `${resumed.pre} vs ${paused.pre}`);
  ok("resume restores the samples", resumed.samples === paused.samples && resumed.samples > 0);

  // ---- 3. train to the end after resuming ----
  await page.click("#btn-train");
  await waitStatus("Finished");
  clock("pretrain (with pause and resume)", t0);
  result.pretrainDone = await status();
  result.pretrainSpeed = await text("#s-speed");
  ok("workers stayed identical", workers === "1" || result.pretrainDone.includes("identical models"), result.pretrainDone);
  const pre = await state();
  ok("pre-training loss fell", pre.lastPre.val < 4, JSON.stringify(pre.lastPre));
  ok("pre-training chart has two lines", (await page.locator("#chart path.line").count()) === 2);

  // The hover markers must be truly invisible until the pointer is over the chart
  // (an SVG <g hidden> is NOT hidden by default: it once left a stray dot in the corner).
  const hoverDisplay = () => page.evaluate(() => getComputedStyle(document.querySelector("#hover")).display);
  ok("hover markers are hidden at rest", (await hoverDisplay()) === "none");
  await page.locator("#chart").scrollIntoViewIfNeeded();
  const box = await page.locator("#chart").boundingBox();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
  await page.waitForTimeout(100);
  result.tooltip = (await page.locator("#tooltip").isVisible()) ? await page.locator("#tooltip").innerText() : null;
  ok("hover shows a tooltip and the markers", !!result.tooltip && (await hoverDisplay()) !== "none", String(result.tooltip));
  await page.mouse.move(0, 0);
  await page.waitForTimeout(100);
  ok("hover markers hide again", (await hoverDisplay()) === "none");

  // ---- 4. fine-tune ----
  ok("fine-tune button is enabled", await page.locator("#btn-ft").isEnabled());
  await page.fill("#ft-steps", ftSteps);
  t0 = Date.now();
  await page.click("#btn-ft");
  await waitStatus("Fine-tuning finished");
  clock("fine-tune", t0);
  result.ftDone = await status();
  const ft = await state();
  ok("phase is now finetune", ft.phase === "finetune" && ft.template?.includes("{}"), JSON.stringify(ft));
  ok("fine-tune chart has points", ft.ft > 2, `${ft.ft}`);
  result.ftTable = {
    word: [await text("#ft-before-word"), await text("#ft-after-word")],
    control: [await text("#ft-before-chance"), await text("#ft-after-chance")],
    stops: [await text("#ft-before-stops"), await text("#ft-after-stops")],
  };
  const num = (s) => Number(s.replace("%", ""));
  ok("fine-tuning raised 'uses the word'", num(result.ftTable.word[1]) > num(result.ftTable.word[0]) + 5, JSON.stringify(result.ftTable.word));
  ok("fine-tuning taught it to stop", num(result.ftTable.stops[1]) > 90, JSON.stringify(result.ftTable.stops));
  ok("fine-tuning workers identical", workers === "1" || result.ftDone.includes("identical models"), result.ftDone);
  await page.locator("#tab-pretrain").click();
  ok("pre-training tab keeps its chart", (await page.locator("#chart path.line").count()) === 2);
  await page.locator("#tab-finetune").click();

  // ---- 5. ask the fine-tuned model ----
  await page.fill("#ask-word", "garden");
  await page.click("#btn-ask");
  await page.waitForFunction(() => document.querySelectorAll("#answers li:not(.empty)").length === 3, null, { timeout: 30000 });
  result.answers = await page.locator("#answers li").allInnerTexts();
  ok("three answers", result.answers.length === 3 && result.answers.every((a) => a.length > 5));
  await page.locator("#chart").scrollIntoViewIfNeeded();
  await page.screenshot({ path: `runs/web-check-final${tag}.png`, fullPage: true });
  result.horizontalOverflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  ok("no horizontal scroll", !result.horizontalOverflow);

  // ---- 6. downloads ----
  for (const [button, file] of [["#btn-dl-model", "model.safetensors"], ["#btn-dl-tok", "tokenizer.json"], ["#btn-dl-sft", "sft.json"]]) {
    const [dl] = await Promise.all([page.waitForEvent("download"), page.click(button)]);
    fs.writeFileSync(`runs/web-export/${file}`, fs.readFileSync(await dl.path()));
    ok(`download ${file}`, dl.suggestedFilename() === file);
  }
  const bytes = fs.readFileSync("runs/web-export/model.safetensors");
  const headerLen = Number(bytes.readBigUInt64LE(0));
  const header = JSON.parse(bytes.subarray(8, 8 + headerLen).toString());
  result.download = { bytes: bytes.length, tensors: Object.keys(header).length - 1, metadata: header.__metadata__ };
  ok("model file is safetensors", headerLen + 8 < bytes.length && result.download.tensors > 10);

  // ---- 7. the native command-line tool understands the browser's output ----
  const tlm = "target/release/tlm.exe";
  if (fs.existsSync(tlm)) {
    result.cliInfo = execFileSync(tlm, ["info", "--model", "runs/web-export"], { encoding: "utf8" }).trim().split("\n");
    result.cliChat = execFileSync(tlm, ["chat", "--model", "runs/web-export", "--input", "garden", "--n", "2"], { encoding: "utf8" }).trim().split("\n");
    ok("tlm chat reads the browser's fine-tuned model", result.cliChat.length === 2 && result.cliChat.every((l) => l.length > 5), JSON.stringify(result.cliChat));
  } else {
    result.cliChat = "skipped: build the CLI first";
  }

  // ---- 8. reload: the fine-tuned session resumes with everything restored ----
  await page.waitForFunction(() => document.querySelector("#save-status").textContent.startsWith("Saved"), null, { timeout: 30000 });
  await page.reload();
  await page.waitForSelector("#resume-banner:not([hidden])", { timeout: 20000 });
  await page.click("#btn-resume");
  await waitStatus("Resumed at step", 120000);
  const back = await state();
  ok("fine-tuned session resumes", back.phase === "finetune" && back.pre > 2 && back.ft > 2 && back.template, JSON.stringify(back));
  ok("results table restored", num(await text("#ft-after-word")) === num(result.ftTable.word[1]));
  await page.fill("#ask-word", "garden");
  await page.click("#btn-ask");
  await page.waitForFunction(() => document.querySelectorAll("#answers li:not(.empty)").length === 3, null, { timeout: 30000 });
  ok("resumed model still answers", true);

  // ---- 8b. preference tuning: mine pairs across the workers, train, pause, reload, resume ----
  const dpoSteps = process.env.E2E_DPO_STEPS || "600"; // long enough to pause part-way: a short run finishes in ~2 s
  ok("preference button is enabled once fine-tuned", await page.locator("#btn-dpo").isEnabled());
  await page.fill("#dpo-prompts", process.env.E2E_DPO_PROMPTS || "300");
  await page.fill("#dpo-steps", dpoSteps);
  t0 = Date.now();
  await page.click("#btn-dpo");
  await page.waitForFunction(() => JSON.parse(document.body.dataset.tlb).phase === "dpo", null, { timeout: 280000 });
  clock("mine pairs and start preference tuning", t0);
  const dpoStart = await state();
  result.dpoInfo = dpoStart.dpoInfo;
  ok("pairs were mined", !!dpoStart.dpoInfo && dpoStart.dpoInfo.pairs >= 40, JSON.stringify(dpoStart.dpoInfo));
  result.dpoNote = await text("#dpo-note");
  ok("before-numbers shown at the start", (await text("#dpo-before-word")).endsWith("%"));

  await page.waitForFunction((n) => JSON.parse(document.body.dataset.tlb).step >= n, Math.round(Number(dpoSteps) * 0.4), { timeout: 200000 });
  await page.click("#btn-pause");
  await page.waitForFunction(() => !document.querySelector("#btn-train").disabled, null, { timeout: 60000 });
  const dpoPaused = await state();
  ok("paused part-way through preference tuning", dpoPaused.phase === "dpo" && dpoPaused.step > 0 && dpoPaused.step < Number(dpoSteps), JSON.stringify(dpoPaused));
  await page.waitForFunction(() => document.querySelector("#save-status").textContent.includes("preference tuning"), null, { timeout: 60000 });

  await page.reload();
  await page.waitForSelector("#resume-banner:not([hidden])", { timeout: 20000 });
  result.dpoBanner = await text("#resume-text");
  await page.click("#btn-resume");
  await waitStatus("Resumed at step", 120000);
  const dpoBack = await state();
  ok("preference tuning resumes in the right phase and step", dpoBack.phase === "dpo" && dpoBack.step === dpoPaused.step, JSON.stringify(dpoBack));
  ok("resume restores the pair statistics", dpoBack.dpoInfo?.pairs === dpoStart.dpoInfo.pairs, JSON.stringify(dpoBack.dpoInfo));
  ok("resume restores the earlier chart history", dpoBack.pre > 2 && dpoBack.ft > 2 && dpoBack.dpoPoints >= 1);

  await page.click("#btn-train");
  await waitStatus("Preference tuning finished");
  result.dpoDone = await status();
  ok("preference tuning workers identical", workers === "1" || result.dpoDone.includes("identical models"), result.dpoDone);
  const dpoEnd = await state();
  ok("preference chart has points", dpoEnd.dpoPoints > 2, `${dpoEnd.dpoPoints}`);
  ok("held-out preference accuracy beats chance", dpoEnd.dpoAccuracy > 0.5, `${dpoEnd.dpoAccuracy}`);
  ok("preference loss fell below ln 2", dpoEnd.lastDpo.val < Math.LN2, `${dpoEnd.lastDpo.val}`);
  result.dpoTable = {
    word: [await text("#dpo-before-word"), await text("#dpo-after-word")],
    control: [await text("#dpo-before-chance"), await text("#dpo-after-chance")],
    stops: [await text("#dpo-before-stops"), await text("#dpo-after-stops")],
  };
  ok("results table filled in", result.dpoTable.word.every((v) => v.endsWith("%")), JSON.stringify(result.dpoTable));
  ok("the fine-tuning results are still there", (await text("#ft-after-word")).endsWith("%"));
  await page.locator("#tab-finetune").click();
  ok("fine-tuning tab keeps its chart after preference tuning", (await page.locator("#chart path.line").count()) === 2);
  await page.locator("#tab-dpo").click();
  await page.fill("#ask-word", "garden");
  await page.click("#btn-ask");
  await page.waitForFunction(() => document.querySelectorAll("#answers li:not(.empty)").length === 3, null, { timeout: 30000 });
  ok("the preference-tuned model answers", true);
  await page.locator("#chart").scrollIntoViewIfNeeded();
  await page.screenshot({ path: `runs/web-check-dpo${tag}.png`, fullPage: true });
  ok("no horizontal scroll with the preference panel", !(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)));
  await page.locator(".dpo").screenshot({ path: `runs/web-check-dpo-panel${tag}.png` });

  // The exported model records every phase's steps, and the native tool can use it.
  const [dlDpo] = await Promise.all([page.waitForEvent("download"), page.click("#btn-dl-model")]);
  fs.mkdirSync("runs/web-export-dpo", { recursive: true });
  fs.writeFileSync("runs/web-export-dpo/model.safetensors", fs.readFileSync(await dlDpo.path()));
  for (const [button, file] of [["#btn-dl-tok", "tokenizer.json"], ["#btn-dl-sft", "sft.json"]]) {
    const [d] = await Promise.all([page.waitForEvent("download"), page.click(button)]);
    fs.writeFileSync(`runs/web-export-dpo/${file}`, fs.readFileSync(await d.path()));
  }
  if (fs.existsSync("target/release/tlm.exe")) {
    const info = execFileSync("target/release/tlm.exe", ["info", "--model", "runs/web-export-dpo"], { encoding: "utf8" });
    result.cliInfoDpo = info.trim().split("\n")[0];
    const expected = Number(steps) + Number(ftSteps) + Number(dpoSteps);
    ok("exported model counts steps from all three phases", info.includes(`trained steps: ${expected}`), `${result.cliInfoDpo} (expected ${expected})`);
  }
  // ---- 9. open the downloaded files in a clean page state ----
  await page.reload();
  await page.waitForSelector("#btn-open");
  await page.setInputFiles("#open-files", ["runs/web-export/model.safetensors", "runs/web-export/tokenizer.json", "runs/web-export/sft.json"]);
  await page.click("#btn-open");
  await waitStatus("Opened a model", 60000);
  const opened = await state();
  ok("opened model is in the loaded phase", opened.phase === "loaded" && opened.template, JSON.stringify(opened));
  ok("opened model cannot train", await page.locator("#btn-train").isDisabled() && await page.locator("#btn-ft").isDisabled());
  await page.fill("#ask-word", "garden");
  await page.click("#btn-ask");
  await page.waitForFunction(() => document.querySelectorAll("#answers li:not(.empty)").length === 3, null, { timeout: 30000 });
  result.openedAnswers = await page.locator("#answers li").allInnerTexts();
  ok("opened model answers questions", result.openedAnswers.length === 3);
  await page.click("#btn-gen");
  await page.waitForFunction(() => document.querySelector("#output").textContent.length > 30, null, { timeout: 30000 });
  ok("opened model continues text", true);

  // ---- 10. one worker vs several: the same training, so about the same loss ----
  if (workers !== "1") {
    await page.reload();
    await page.waitForSelector("#btn-load");
    t0 = Date.now();
    await prepare("1");
    await page.click("#btn-train");
    await waitStatus("Finished");
    clock("pretrain with 1 worker", t0);
    result.singleSpeed = await text("#s-speed");
    const single = await state();
    result.valSingle = single.lastPre.val;
    result.valParallel = pre.lastPre.val;
    ok("1 worker and several workers reach similar loss", Math.abs(single.lastPre.val - pre.lastPre.val) < 0.2, `${single.lastPre.val} vs ${pre.lastPre.val}`);
  }

  return result;
}
