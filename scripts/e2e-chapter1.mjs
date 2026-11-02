// End-to-end test for the textbook chapter 1, run with scripts/run-e2e.mjs:
//   node scripts/run-e2e.mjs http://localhost:8080/textbook/chapter1.html scripts/e2e-chapter1.mjs
// Needs: .\scripts\build-web.ps1 done and `python -m http.server 8080 --directory web` running.
// Environment: E2E_DARK / E2E_MOBILE for visual variants, E2E_TAG for the screenshot suffix.

export default async function run(page) {
  const result = { checks: {} };
  const ok = (name, pass, detail) => (result.checks[name] = pass ? "pass" : `FAIL ${detail ?? ""}`);
  const tag = process.env.E2E_TAG || "";
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));

  if (process.env.E2E_DARK) await page.emulateMedia({ colorScheme: "dark" });
  if (process.env.E2E_MOBILE) await page.setViewportSize({ width: 390, height: 800 });

  const state = () => page.evaluate(() => JSON.parse(document.body.dataset.book ?? "null"));
  await page.waitForFunction(() => !document.querySelector("#k").disabled, null, { timeout: 40000 });
  ok("status says ready", (await page.textContent("#lab-status")).startsWith("Ready."), await page.textContent("#lab-status"));

  const text = "Once upon a time, a little girl named Lily found a shiny red ball in the garden.";
  await page.fill("#text", text);
  const bytes = new TextEncoder().encode(text).length;

  // k = 0: one token per byte, vocabulary 256
  let s = await state();
  ok("k=0 gives one token per byte", s.k === 0 && s.vocab === 256 && s.tokens === bytes, JSON.stringify(s));
  ok("k=0 chips match", (await page.locator(".chips .chip").count()) === bytes);

  // slide to the end: fewer tokens, round trip holds
  const max = Number(await page.getAttribute("#k", "max"));
  ok("slider range is the learned merges", max >= 256, String(max));
  await page.evaluate((m) => { const k = document.querySelector("#k"); k.value = m; k.dispatchEvent(new Event("input")); }, max);
  s = await state();
  ok("full merges shrink the token list", s.vocab === 256 + max && s.tokens < bytes / 2, JSON.stringify(s));
  ok("round trip is exact", (await page.textContent("#roundtrip")).includes("exactly"), await page.textContent("#roundtrip"));
  ok("recipe highlights exactly the latest merge", (await page.locator("#recipe tr.current").count()) === 1);
  ok("recipe lists every learned merge", (await page.locator("#recipe tr").count()) === max);
  ok("latest merge is scrolled into view", await page.evaluate(() => {
    const w = document.querySelector(".table-wrap"), r = document.querySelector("#recipe tr.current");
    const a = w.getBoundingClientRect(), b = r.getBoundingClientRect();
    return b.top >= a.top - 1 && b.bottom <= a.bottom + 1;
  }));

  // intermediate point: monotone
  await page.evaluate(() => { const k = document.querySelector("#k"); k.value = 100; k.dispatchEvent(new Event("input")); });
  const mid = await state();
  ok("tokens decrease monotonically", mid.tokens > s.tokens && mid.tokens < bytes, `${mid.tokens} vs ${s.tokens}`);
  ok("latest-merge line shows merge 100", (await page.textContent("#latest")).startsWith("Merge 100:"), await page.textContent("#latest"));

  // non-English text and emoji fall back to bytes and still round-trip
  await page.fill("#text", "héllo 🌍 日本");
  const e = await state();
  ok("unicode round-trips", (await page.textContent("#roundtrip")).includes("exactly"), await page.textContent("#roundtrip"));
  ok("emoji shows raw byte chips", (await page.locator(".chip.raw").count()) > 0);

  // empty text
  await page.fill("#text", "");
  const z = await state();
  ok("empty text has zero tokens", z.tokens === 0);

  await page.fill("#text", text);
  await page.evaluate(() => { const k = document.querySelector("#k"); k.value = 256; k.dispatchEvent(new Event("input")); });
  result.bytes_per_token_at_256_merges = await page.textContent("#s-ratio");
  ok("no page errors", errors.length === 0, errors.join(" | "));

  await page.screenshot({ path: `textbook-ch1${tag}.png`, fullPage: true });
  result.failures = Object.entries(result.checks).filter(([, v]) => v !== "pass").length;
  return result;
}
