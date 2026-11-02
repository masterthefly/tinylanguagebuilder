// Saves one session in the browser's IndexedDB so training can be resumed after
// the tab is closed. Everything stays on this device. Every call is wrapped by the
// caller in try/catch: storage can be blocked (private windows, site settings).

const DB_NAME = "tinylanguagebuilder";
const STORE = "sessions";
const KEY = "autosave";

function open() {
  return new Promise((resolve, reject) => {
    if (!("indexedDB" in globalThis)) return reject(new Error("IndexedDB is not available"));
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("could not open storage"));
    request.onblocked = () => reject(new Error("storage is blocked by another tab"));
  });
}

async function run(mode, action) {
  const db = await open();
  try {
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, mode);
      const request = action(tx.objectStore(STORE));
      tx.oncomplete = () => resolve(request?.result);
      tx.onerror = () => reject(tx.error ?? new Error("storage write failed"));
      tx.onabort = () => reject(tx.error ?? new Error("storage write was aborted (out of space?)"));
    });
  } finally {
    db.close();
  }
}

/** Replace the saved session. `record` is a plain object of strings, numbers and typed arrays. */
export const saveSession = (record) => run("readwrite", (s) => s.put(record, KEY));

/** The saved session, or undefined if there is none. */
export const loadSession = () => run("readonly", (s) => s.get(KEY));

export const deleteSession = () => run("readwrite", (s) => s.delete(KEY));
