import test from "node:test";
import assert from "node:assert/strict";
import { fetchJson } from "../src/apiClient.js";

test("fetchJson returns decoded success and turns API detail into a clear error", async () => {
  assert.deepEqual(await fetchJson("/ok", {}, 100, async () => ({ ok: true, json: async () => ({ value: 1 }) })), { value: 1 });
  await assert.rejects(fetchJson("/bad", {}, 100, async () => ({ ok: false, status: 503, json: async () => ({ detail: { code: "offline", message: "Provider offline" } }) })), /Provider offline/);
});
test("fetchJson reports timeout and malformed JSON", async () => {
  await assert.rejects(fetchJson("/slow", {}, 5, (_url, options) => new Promise((_resolve, reject) => options.signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError"))))), /timed out/);
  await assert.rejects(fetchJson("/bad-json", {}, 100, async () => ({ ok: true, json: async () => { throw new Error("bad json"); } })), /unreadable response/);
});
