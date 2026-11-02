// End-to-end test for the textbook's "Explain" popover (select text, press Explain, read the glossary entry),
// run with scripts/run-e2e.mjs:
//   node scripts/run-e2e.mjs http://localhost:8080/textbook/chapter3.html scripts/e2e-explain.mjs
// Needs: `python -m http.server 8080 --directory web` running (no wasm build needed: the popover is plain JavaScript).
// Environment: E2E_DARK / E2E_MOBILE for visual variants, E2E_TAG for the screenshot suffix.

export default async function run(page) {
  const tag = process.env.E2E_TAG || "";
  const result = { checks: {}, info: {} };
  const ok = (name, pass, detail) => (result.checks[name] = pass ? "pass" : `FAIL ${detail ?? ""}`);
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));

  if (process.env.E2E_DARK) await page.emulateMedia({ colorScheme: "dark" });
  if (process.env.E2E_MOBILE) await page.setViewportSize({ width: 390, height: 800 });

  const explain = () => page.evaluate(() => JSON.parse(document.body.dataset.explain ?? "null"));
  const btnVisible = () => page.evaluate(() => { const b = document.getElementById("explain-btn"); return !!b && !b.hidden; });
  const panelVisible = () => page.evaluate(() => { const p = document.getElementById("explain-panel"); return !!p && !p.hidden; });
  const panelText = () => page.evaluate(() => document.getElementById("explain-panel").innerText);
  const settle = () => page.waitForTimeout(300);

  /** Select the first occurrence of `needle` inside main.book, like dragging the mouse over it. */
  const select = (needle, within = "main.book") => page.evaluate(([needle, within]) => {
    const root = document.querySelector(within);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const n = walker.currentNode;
      const i = n.data.indexOf(needle);
      if (i >= 0) {
        const r = document.createRange();
        r.setStart(n, i);
        r.setEnd(n, i + needle.length);
        const s = getSelection();
        s.removeAllRanges();
        s.addRange(r);
        n.parentElement.scrollIntoView({ block: "center" });
        return true;
      }
    }
    return false;
  }, [needle, within]);
  const clearSelection = () => page.evaluate(() => getSelection().removeAllRanges());

  // ---- the glossary itself ----
  const g = await page.evaluate(async () => {
    const { GLOSSARY, CHAPTERS } = await import(new URL("glossary.js", location.href).href);
    const { lookup } = await import(new URL("explain.js", location.href).href);    // already loaded by the page: same instance
    const ids = GLOSSARY.map((e) => e.id);
    return {
      count: GLOSSARY.length,
      uniqueIds: new Set(ids).size === ids.length,
      badRelated: GLOSSARY.flatMap((e) => (e.related ?? []).filter((r) => !ids.includes(r)).map((r) => `${e.id}->${r}`)),
      badChapter: GLOSSARY.filter((e) => e.chapter && !CHAPTERS[e.chapter[0]]).map((e) => e.id),
      missingText: GLOSSARY.filter((e) => !e.term || !e.short || !e.aliases?.length).map((e) => e.id),
      termNotFirst: GLOSSARY.filter((e) => lookup(e.term)[0]?.id !== e.id).map((e) => `${e.id} -> ${lookup(e.term)[0]?.id}`),
      aliasMissing: GLOSSARY.flatMap((e) => e.aliases.filter((a) => !lookup(a).some((x) => x.id === e.id)).map((a) => `${e.id}:${a}`)),
      plural: lookup("softmaxes")[0]?.id === "softmax" && lookup("tokens")[0]?.id === "token" && lookup("Learning Rates")[0]?.id === "learning-rate",
      hyphenSpace: lookup("byte pair encoding")[0]?.id === "bpe" && lookup("top k")[0]?.id === "top-k" && lookup("fine tuning")[0]?.id === "fine-tuning",
      noSubword: lookup("keyboard").length === 0 && lookup("the unkeyed lock").length === 0,
      empty: lookup("").length === 0,
    };
  });
  ok("glossary has at least 40 entries", g.count >= 40, String(g.count));
  ok("glossary ids are unique", g.uniqueIds);
  ok("every related id exists", g.badRelated.length === 0, g.badRelated.join(", "));
  ok("every chapter number is a real chapter", g.badChapter.length === 0, g.badChapter.join(", "));
  ok("every entry has a term, an answer and aliases", g.missingText.length === 0, g.missingText.join(", "));
  ok("selecting an entry's own name gives that entry first", g.termNotFirst.length === 0, g.termNotFirst.join(", "));
  ok("every alias finds its entry", g.aliasMissing.length === 0, g.aliasMissing.join(", "));
  ok("plurals and capitals match", g.plural);
  ok("hyphen and space are interchangeable", g.hyphenSpace);
  ok("words are matched whole (no 'key' inside 'keyboard')", g.noSubword);
  ok("empty selection finds nothing", g.empty);

  // ---- nothing shown until something is selected ----
  await settle();
  ok("no button or panel at first", !(await btnVisible()) && !(await panelVisible()));

  // ---- a single term ----
  ok("found the word to select", await select("softmax"));
  await page.waitForFunction(() => !document.getElementById("explain-btn").hidden, null, { timeout: 3000 });
  const box = await page.locator("#explain-btn").boundingBox();
  const vp = page.viewportSize();
  ok("button appears next to the selection, inside the window", box && box.x >= 0 && box.y >= 0 && box.x + box.width <= vp.width && box.y + box.height <= vp.height, JSON.stringify(box));
  ok("button is labelled", (await page.locator("#explain-btn").textContent()) === "Explain");
  await page.click("#explain-btn");
  ok("clicking Explain opens the panel and hides the button", (await panelVisible()) && !(await btnVisible()));
  let e = await explain();
  ok("softmax is found first", e.matches[0] === "softmax", JSON.stringify(e));
  ok("the panel quotes the selection", e.selection === "softmax");
  let text = await panelText();
  ok("the panel explains it", /probabilities/.test(text) && /Softmax/.test(text), text.slice(0, 200));
  ok("the panel says where it came from", /From: Chapter 3/.test(text), text.slice(0, 200));
  ok("the panel says it is covered in this chapter", /Covered in this chapter/.test(text));
  const pbox = await page.locator("#explain-panel").boundingBox();
  ok("the panel fits inside the window", pbox && pbox.x >= 0 && pbox.y >= 0 && pbox.x + pbox.width <= vp.width + 1 && pbox.y + pbox.height <= vp.height + 1, JSON.stringify(pbox));
  await page.screenshot({ path: `textbook-explain${tag}.png` });

  // ---- related terms ----
  const chipName = await page.locator("#explain-panel .explain-card .explain-chip").first().textContent();
  await page.locator("#explain-panel .explain-card .explain-chip").first().click();
  e = await explain();
  ok("a related-term chip shows that term", e.matches.length === 1 && (await panelText()).includes(chipName), JSON.stringify({ chipName, e }));

  // ---- closing ----
  await page.keyboard.press("Escape");
  ok("Escape closes the panel", !(await panelVisible()));
  ok("and clears the selection", (await page.evaluate(() => getSelection().toString())) === "");

  // ---- phrases, headings, plurals ----
  await select("causal mask");
  await settle();
  await page.click("#explain-btn");
  ok("a two-word term is found (causal mask)", (await explain()).matches[0] === "causal-mask");
  await page.keyboard.press("Escape");

  await select("Query, key, value");
  await settle();
  await page.click("#explain-btn");
  e = await explain();
  ok("a heading selection finds query/key/value first", e.matches[0] === "qkv", JSON.stringify(e.matches));
  ok("the section is named", e.where.includes("Query, key, value"), e.where);
  await page.keyboard.press("Escape");

  // ---- a term from another chapter links to it ----
  await select("embedding");
  await settle();
  await page.click("#explain-btn");
  ok("an embedding selection finds the embedding entry", (await explain()).matches[0] === "embedding");
  ok("and links to chapter 2", (await page.locator('#explain-panel a[href="chapter2.html"]').count()) >= 1);
  await page.keyboard.press("Escape");

  // ---- a whole passage ----
  const passage = await page.evaluate(() => document.querySelector("#h-qkv").parentElement.querySelector("p").textContent.trim());
  await select(passage.slice(0, 120));
  await settle();
  await page.click("#explain-btn");
  e = await explain();
  text = await panelText();
  ok("a long selection lists several terms", e.matches.length >= 2 && /Terms found in your selection/.test(text), JSON.stringify(e.matches));
  ok("and shows at most five cards", (await page.locator("#explain-panel .explain-card").count()) <= 5);
  await page.keyboard.press("Escape");

  // ---- no match ----
  await select("Lily");
  await settle();
  await page.click("#explain-btn");
  ok("an unknown word says so politely", (await explain()).matches.length === 0 && /No glossary entry/.test(await panelText()));
  ok("and still offers every term", (await page.locator("#explain-panel .explain-all .explain-chip").count()) >= 40);
  await page.locator("#explain-panel .explain-all summary").click();
  await page.locator("#explain-panel .explain-all .explain-chip", { hasText: "Temperature" }).first().click();
  ok("a term from the list opens its entry", (await explain()).matches[0] === "temperature" && /Covered in chapter 5/.test(await panelText().then((t) => t.replace(/\s+/g, " "))) || /chapter 5: Sampling/.test(await panelText()));
  await page.keyboard.press("Escape");

  // ---- keyboard shortcut ----
  await select("attention head");
  await page.keyboard.press("Alt+e");
  ok("Alt+E opens the panel for the current selection", (await panelVisible()) && (await explain()).matches[0] === "head");
  await page.keyboard.press("Escape");

  // ---- where it must NOT appear ----
  ok("found the page title", await select("Chapter 3: Attention", "header"));
  await settle();
  ok("selecting the page header shows no button", !(await btnVisible()));
  await clearSelection();
  await select("a");
  await clearSelection();
  await settle();
  ok("clearing the selection hides the button", !(await btnVisible()));
  await select("a", "main.book");
  await settle();
  ok("a one-letter selection is ignored", !(await btnVisible()));
  await clearSelection();

  // ---- clicking elsewhere closes it ----
  await select("softmax");
  await settle();
  await page.click("#explain-btn");
  await page.mouse.click(5, 5);
  ok("clicking elsewhere closes the panel", !(await panelVisible()));

  // ---- the contents page has it too ----
  await page.goto(new URL("./", page.url()).href);
  await select("embeddings", "main.book").catch(() => {});
  await select("Embeddings");
  await settle();
  ok("the contents page offers Explain too", await btnVisible());

  ok("no page errors", errors.length === 0, errors.join(" | "));
  result.failures = Object.entries(result.checks).filter(([, v]) => v !== "pass").length;
  return result;
}
