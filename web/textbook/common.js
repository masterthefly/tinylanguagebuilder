// Helpers shared by the textbook chapters.

export const $ = (id) => document.getElementById(id);

const strict = new TextDecoder("utf-8", { fatal: true });

/** Show a token's bytes as text: spaces as a dot, newlines as an arrow, broken UTF-8 as \xNN. */
export function label(bytes) {
  let text;
  try {
    text = strict.decode(new Uint8Array(bytes));
  } catch {
    return { text: bytes.map((b) => "\\x" + b.toString(16).padStart(2, "0")).join(""), raw: true };
  }
  // Control characters (NUL, tab, ...) would draw as empty boxes, so show them as \xNN like raw bytes.
  const spaced = text.replace(/ /g, "·").replace(/\n/g, "↵");
  const shown = spaced.replace(/[\x00-\x1f\x7f]/g, (c) => "\\x" + c.charCodeAt(0).toString(16).padStart(2, "0"));
  return { text: shown, raw: shown !== spaced };
}

/** A small coloured chip showing one token. `cls` picks the tint (c0, c1, c2). */
export function chip(bytes, cls = "c2") {
  const l = label(bytes);
  const el = document.createElement("span");
  el.className = `chip ${cls}${l.raw ? " raw" : ""}`;
  el.textContent = l.text;
  return el;
}

/** The token's text if its bytes are valid UTF-8, otherwise null. */
export function tokenString(bytes) {
  try {
    return strict.decode(new Uint8Array(bytes));
  } catch {
    return null;
  }
}

/** Fetch the TinyStories sample the app also uses. Returns null if it is not there. */
export async function fetchSample() {
  try {
    const r = await fetch("../data/tinystories-8mb.txt");
    return r.ok ? await r.text() : null;
  } catch {
    return null;
  }
}

/** Wait for the browser to paint, so a status message shows before heavy work starts. */
export const nextPaint = () => new Promise((res) => requestAnimationFrame(() => setTimeout(res, 0)));
