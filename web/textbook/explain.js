// "Explain" for the textbook: select any text on a page and a small button appears; press it (or Alt+E) to get an
// explanation from the built-in glossary. Nothing is sent anywhere: this runs entirely in the page, and it is a
// glossary lookup, not an AI model.
import { GLOSSARY, CHAPTERS } from "./glossary.js";

const MIN_CHARS = 2;
const MAX_CHARS = 1500;                 // longer selections are not worth searching
const MAX_CARDS = 3;
const MAX_CARDS_LONG = 5;
const LONG_SELECTION = 60;              // above this many characters a selection is treated as a passage, not a term

const byId = new Map(GLOSSARY.map((e) => [e.id, e]));
const thisChapter = Number(location.pathname.match(/chapter(\d+)\.html/)?.[1] ?? 0);

// ---- finding entries ----------------------------------------------------------

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** One regex per alias: word boundaries, optional plural ending, and "-" / space treated alike. */
const matchers = GLOSSARY.map((entry) => ({
  entry,
  aliases: entry.aliases.map((alias) => ({
    alias,
    re: new RegExp(`(?<![a-z0-9_])${alias.toLowerCase().split(/[\s-]+/).map(escapeRe).join("[\\s-]+")}(?:s|es)?(?![a-z0-9_])`),
  })),
}));

const clean = (s) => s.toLowerCase().replace(/[‘’]/g, "'").replace(/\s+/g, " ").trim();
const bare = (s) => clean(s).replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, "");

/** Glossary entries for a selection, best first. An entry that IS the selection beats one that is merely inside it. */
export function lookup(selection) {
  const text = clean(selection);
  const whole = bare(selection);
  const found = [];
  matchers.forEach(({ entry, aliases }, order) => {
    let best = -1;
    for (const { alias, re } of aliases) {
      if (!re.test(text)) continue;
      const a = alias.toLowerCase();
      const exact = whole === a || whole === a + "s" || whole === a + "es" || whole === bare(entry.term);
      best = Math.max(best, exact ? 1000 + a.length : a.length);
    }
    if (best >= 0) found.push({ entry, score: best, order });
  });
  found.sort((a, b) => b.score - a.score || a.order - b.order);
  return found.map((f) => f.entry);
}

// ---- the page furniture -------------------------------------------------------

let button = null;
let panel = null;
let lastText = "";
let lastWhere = "";

function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  e.append(...kids);
  return e;
}

function buildFurniture() {
  button = el("button", { type: "button", class: "explain-btn primary", id: "explain-btn", "aria-label": "Explain the selected text" }, "Explain");
  button.hidden = true;
  // Pressing the button must not clear the selection it is about to explain.
  button.addEventListener("mousedown", (e) => e.preventDefault());
  button.addEventListener("click", () => openPanel());
  panel = el("aside", { class: "explain-panel", id: "explain-panel", role: "dialog", "aria-label": "Explanation of the selected text", tabindex: "-1" });
  panel.hidden = true;
  document.body.append(button, panel);
}

const inBook = (node) => !!(node && (node.nodeType === 1 ? node : node.parentElement)?.closest?.("main.book"));
const inOurUi = (node) => !!(node && (node.nodeType === 1 ? node : node.parentElement)?.closest?.(".explain-panel, .explain-btn"));

/** The heading the selection sits under, for "From: ..." */
function sectionOf(range) {
  let node = range.startContainer.nodeType === 1 ? range.startContainer : range.startContainer.parentElement;
  const section = node?.closest("section");
  const h = section?.querySelector("h2");
  return h ? h.textContent.trim() : "";
}

function currentSelection() {
  const sel = window.getSelection();
  if (!sel || sel.rangeCount === 0 || sel.isCollapsed) return null;
  const range = sel.getRangeAt(0);
  if (!inBook(range.commonAncestorContainer) || inOurUi(range.commonAncestorContainer)) return null;
  const text = sel.toString().replace(/\s+/g, " ").trim();
  if (text.length < MIN_CHARS || text.length > MAX_CHARS) return null;
  return { text, range };
}

let timer = 0;
function onSelectionChange() {
  clearTimeout(timer);
  timer = setTimeout(() => {
    const s = currentSelection();
    if (!s || !panel.hidden) {
      button.hidden = true;
      return;
    }
    placeButton(s.range);
  }, 120);
}

function placeButton(range) {
  const rects = range.getClientRects();
  const r = rects.length ? rects[rects.length - 1] : range.getBoundingClientRect();
  button.hidden = false;
  const w = button.offsetWidth || 80;
  const left = Math.max(8, Math.min(window.innerWidth - w - 8, r.right - w / 2));
  // Below the selection's end, so it stays clear of the browser's own touch toolbar above it.
  const top = Math.min(window.innerHeight - 44, r.bottom + 10);
  button.style.left = `${left}px`;
  button.style.top = `${top}px`;
}

// ---- the panel ----------------------------------------------------------------

function openPanel() {
  const s = currentSelection();
  if (!s) return;
  lastText = s.text;
  lastWhere = sectionOf(s.range);
  button.hidden = true;
  renderResults(lookup(s.text), s.text);
  panel.hidden = false;
  panel.focus({ preventScroll: true });
}

function closePanel() {
  if (panel.hidden) return;
  panel.hidden = true;
  window.getSelection()?.removeAllRanges();
}

function header(selectionText) {
  const quote = selectionText.length > 140 ? selectionText.slice(0, 137) + "…" : selectionText;
  const where = lastWhere ? `${thisChapter ? `Chapter ${thisChapter} › ` : ""}${lastWhere}` : thisChapter ? `Chapter ${thisChapter}` : "";
  return el("div", { class: "explain-head" },
    el("div", {},
      el("p", { class: "explain-kicker" }, "You selected"),
      el("blockquote", { class: "explain-quote" }, quote),
      where ? el("p", { class: "explain-where" }, `From: ${where}`) : "",
    ),
    el("button", { type: "button", class: "explain-close", "aria-label": "Close explanation", onclick: closePanel }, "×"),
  );
}

function card(entry, { open }) {
  const c = el("article", { class: "explain-card" });
  c.append(el("h3", {}, entry.term), el("p", {}, entry.short));
  if (entry.more) c.append(el("p", { class: "explain-more" }, entry.more));
  if (entry.chapter) {
    const [n, title] = entry.chapter;
    c.append(n === thisChapter
      ? el("p", { class: "explain-chapter" }, `Covered in this chapter (${n}. ${title}).`)
      : el("p", { class: "explain-chapter" }, "Covered in ", el("a", { href: CHAPTERS[n] }, `chapter ${n}: ${title}`), "."));
  } else {
    c.append(el("p", { class: "explain-chapter" }, "Not covered in the textbook chapters; it appears in the main app."));
  }
  const related = (entry.related ?? []).map((id) => byId.get(id)).filter(Boolean);
  if (related.length) {
    const row = el("p", { class: "explain-related" }, "Related: ");
    related.forEach((r) => row.append(el("button", { type: "button", class: "explain-chip", onclick: () => showEntry(r) }, r.term)));
    c.append(row);
  }
  if (!open) c.classList.add("secondary");
  return c;
}

function allTerms() {
  const d = el("details", { class: "explain-all" }, el("summary", {}, "Browse all terms"));
  const list = el("p", { class: "explain-related" });
  [...GLOSSARY].sort((a, b) => a.term.localeCompare(b.term)).forEach((e) =>
    list.append(el("button", { type: "button", class: "explain-chip", onclick: () => showEntry(e) }, e.term)));
  d.append(list);
  return d;
}

function renderResults(entries, selectionText) {
  panel.replaceChildren(header(selectionText));
  const body = el("div", { class: "explain-body", "aria-live": "polite" });
  const long = selectionText.length > LONG_SELECTION;
  const shown = entries.slice(0, long ? MAX_CARDS_LONG : MAX_CARDS);
  if (!shown.length) {
    body.append(el("p", { class: "explain-none" },
      "No glossary entry matches this selection. The glossary covers the textbook's technical terms, so try selecting a single term such as ",
      el("em", {}, "softmax"), " or ", el("em", {}, "temperature"), ", or a phrase that contains one."));
  } else {
    if (long) body.append(el("p", { class: "explain-lead" }, `Terms found in your selection (${entries.length}${entries.length > shown.length ? `, showing the ${shown.length} most specific` : ""}):`));
    else if (entries.length > 1 && shown.length > 1) body.append(el("p", { class: "explain-lead" }, "Best match first; the others also appear in your selection."));
    shown.forEach((e, i) => body.append(card(e, { open: i === 0 })));
  }
  body.append(allTerms());
  panel.append(body);
  panel.scrollTop = 0;
  publish(entries, selectionText);
}

/** Show one entry (from a "Related" chip or the term list) without needing a new selection. */
function showEntry(entry) {
  panel.replaceChildren(
    el("div", { class: "explain-head" },
      el("div", {}, el("p", { class: "explain-kicker" }, "Glossary"), el("p", { class: "explain-quote-plain" }, entry.term)),
      el("button", { type: "button", class: "explain-close", "aria-label": "Close explanation", onclick: closePanel }, "×")),
    el("div", { class: "explain-body", "aria-live": "polite" }, card(entry, { open: true }), allTerms()),
  );
  panel.scrollTop = 0;
  publish([entry], entry.term);
}

function publish(entries, selectionText) {
  document.body.dataset.explain = JSON.stringify({ selection: selectionText, matches: entries.map((e) => e.id), where: lastWhere });
}

// ---- start --------------------------------------------------------------------

function main() {
  if (!document.querySelector("main.book")) return;
  if (document.getElementById("explain-btn")) return;      // already set up (the module was loaded twice)
  buildFurniture();
  document.addEventListener("selectionchange", onSelectionChange);
  window.addEventListener("scroll", () => { if (!button.hidden) onSelectionChange(); }, { passive: true });
  window.addEventListener("resize", onSelectionChange);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !panel.hidden) {
      closePanel();
    } else if (e.altKey && (e.key === "e" || e.key === "E")) {
      e.preventDefault();
      openPanel();
    }
  });
  // A click outside the panel and button closes the panel (but a new selection is allowed to start).
  document.addEventListener("mousedown", (e) => {
    if (!panel.hidden && !inOurUi(e.target)) panel.hidden = true;
  });
}
main();
