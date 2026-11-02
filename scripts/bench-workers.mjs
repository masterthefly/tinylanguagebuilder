// How much faster is training with more Web Workers? Run with scripts/run-e2e.mjs:
//   node scripts/run-e2e.mjs http://localhost:8080/ scripts/bench-workers.mjs
// Environment: BENCH_PRESETS (default "small,medium"), BENCH_WORKERS (default "1,2,4,8").

export default async function run(page) {
  const presets = (process.env.BENCH_PRESETS || "small,medium").split(",");
  const workerCounts = (process.env.BENCH_WORKERS || "1,2,4,8").split(",");
  const stepsFor = { tiny: 300, small: 600, medium: 100 };
  const rows = [];

  for (const preset of presets) {
    for (const workers of workerCounts) {
      await page.reload();
      await page.waitForSelector("#btn-load");
      await page.selectOption("#preset", preset);
      await page.fill("#steps", String(stepsFor[preset]));
      await page.selectOption("#workers", workers);
      const t0 = Date.now();
      await page.click("#btn-load");
      await page.waitForFunction(() => !document.querySelector("#btn-train").disabled, null, { timeout: 180000 });
      const prepareSeconds = (Date.now() - t0) / 1000;
      await page.click("#btn-train");
      await page.waitForFunction(() => document.querySelector("#status").textContent.startsWith("Finished"), null, { timeout: 600000 });
      const speed = Number((await page.textContent("#s-speed")).split(" ")[0]);
      const status = await page.textContent("#status");
      rows.push({ preset, workers: Number(workers), stepsPerSec: speed, prepareSeconds, synced: workers === "1" || status.includes("identical") });
    }
  }
  // Speedup relative to one worker, per preset.
  for (const r of rows) {
    const base = rows.find((x) => x.preset === r.preset && x.workers === 1)?.stepsPerSec;
    r.speedup = base ? +(r.stepsPerSec / base).toFixed(2) : null;
  }
  return rows;
}
