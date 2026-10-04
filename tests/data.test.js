import test from "node:test";
import assert from "node:assert/strict";
import { createShardLoader, fetchData, validateDataset } from "../data.js";

test("requests distinguish missing optional data from server failures", async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async () => ({ ok: false, status: 404 });
    assert.equal(await fetchData("missing.bin", "arrayBuffer", true), null);
    await assert.rejects(fetchData("required.bin"), /HTTP 404/);
    globalThis.fetch = async () => ({ ok: false, status: 503 });
    await assert.rejects(fetchData("optional.bin", "arrayBuffer", true), /HTTP 503/);
  } finally { globalThis.fetch = original; }
});

test("shards share in-flight requests and recover after a failed fetch", async () => {
  const original = globalThis.fetch;
  let requests = 0, resolve;
  try {
    globalThis.fetch = () => { requests++; return new Promise(r => { resolve = r; }); };
    const { shardOf, textOf } = createShardLoader(2, 2);
    const first = shardOf(0), second = shardOf(0);
    assert.equal(first, second);
    assert.equal(requests, 1);
    resolve({ ok: false, status: 503 });
    await assert.rejects(first, /503/);
    const retry = textOf(1);
    assert.equal(requests, 2);
    const rows = [0, 1].map(i => ({ text: `turn ${i}`, speaker: "agent", time: "2026-01-01" }));
    resolve({ ok: true, json: async () => rows });
    assert.equal((await retry).text, "turn 1");
    await assert.rejects(textOf(-1), /Invalid turn/);
  } finally { globalThis.fetch = original; }
});

test("mismatched binary files fail before graph construction", () => {
  const meta = { count: 2, shard: 2, days: 1, first: "2026-01-01 00:00:00", last: "2026-01-02 00:00:00", speakers: [{ name: "agent", color: [1, 1, 1] }] };
  const stars = new ArrayBuffer(32), colors = new ArrayBuffer(8), edges = new Uint32Array([0, 1]).buffer;
  validateDataset(meta, stars, colors, edges, null, null, null, null);
  assert.throws(() => validateDataset(meta, stars, colors, edges, new ArrayBuffer(8), null, null, null), /present together/);
  assert.throws(() => validateDataset(meta, stars, colors, new Uint32Array([0, 2]).buffer, null, null, null, null), /invalid turn/);
  assert.throws(() => validateDataset(meta, stars, colors, edges, null, null, edges, new Uint8Array([2]).buffer), /Context kinds/);
});
