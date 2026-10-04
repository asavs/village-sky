import test from "node:test";
import assert from "node:assert/strict";
import vm from "node:vm";
import { readFileSync } from "node:fs";

// Run the actual app's search handlers with controlled network completion order.
const source = readFileSync(new URL("../sky.js", import.meta.url), "utf8");
const searchSource = source.slice(source.indexOf("// Search streams"), source.indexOf("const rgbOf"));
function deferred() { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; }
function harness({ n = 2, textOf = async () => ({ speaker: "agent", time: "2026-01-01" }), shardOf = async () => [] } = {}) {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: "", textContent: "", handlers: {}, classList: { add() {}, remove() {}, contains() { return false; }, toggle() {} },
      addEventListener(name, fn) { this.handlers[name] = fn; }, setAttribute() {}, focus() {}, blur() {}, select() {},
    });
    return elements.get(id);
  };
  let focused;
  const context = vm.createContext({
    n, meta: { shard: n }, document: { getElementById: element }, params: new URLSearchParams(),
    hit: new Float32Array(n), hitAttr: {}, shardOf, textOf, selected: -1,
    speakerColor: () => [1, 1, 1], touched() {}, focus(i) { focused = i; },
    makeRibbons: () => ({ set() {}, show() {} }), fetchData: async () => new ArrayBuffer(n * 96),
    setTimeout, clearTimeout, console: { error() {} },
  });
  vm.runInContext(searchSource, context);
  return { context, element, evaluate: code => vm.runInContext(code, context), get focused() { return focused; } };
}

test("late more-like metadata cannot overwrite a newer query", async () => {
  const text = deferred();
  const h = harness({ textOf: () => text.promise });
  const pending = h.evaluate("moreLike(0)");
  h.element("search").value = "new query";
  await h.evaluate("search('')");
  text.resolve({ speaker: "old agent", time: "2026-01-01" });
  await pending;
  assert.equal(h.element("search").value, "new query");
  assert.equal(h.element("found").textContent, "");
  assert.equal(h.evaluate("hits.length"), 0);
});

test("typing cancels a running shard search before debounce completes", async () => {
  const shard = deferred();
  const h = harness({ shardOf: () => shard.promise });
  const pending = h.evaluate("search('old')");
  h.element("search").value = "new";
  h.element("search").handlers.input();
  shard.resolve([{ speaker: "agent", text: "old" }]);
  await pending;
  assert.equal(h.evaluate("hits.length"), 0);
  assert.equal(h.element("found").textContent, "");
  h.evaluate("resetSearch()"); // cancel the pending timer
});

test("a shard failure stops workers and exposes a recoverable status", async () => {
  const h = harness({ shardOf: async () => { throw new Error("HTTP 503"); } });
  await h.evaluate("search('query')");
  assert.match(h.element("found").textContent, /Try again/);
  assert.equal(h.evaluate("hits.length"), 0);
});

test("Enter handles a query matching the whole 183k-turn dataset", () => {
  const h = harness({ n: 183483 });
  h.evaluate("hits = Array.from({length: n}, (_, i) => n - i - 1)");
  h.element("search").handlers.keydown({ key: "Enter" });
  assert.equal(h.focused, 0);
});

test("nearest results exclude the seed even for a tiny dataset", async () => {
  const h = harness();
  const result = await h.evaluate("nearest(new Float32Array(96), new Set([0]))");
  assert.deepEqual(Array.from(result.order), [1]);
});
