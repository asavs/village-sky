// All data requests check HTTP status; failed cached requests can be retried.
export async function fetchData(url, format = "arrayBuffer", optional = false) {
  const response = await fetch(url);
  if (optional && response.status === 404) return null;
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response[format]();
}

export function createShardLoader(shardSize, count) {
  const shards = new Map();
  const shardOf = s => {
    if (!shards.has(s)) {
      const request = fetchData(`data/text/${String(s).padStart(4, "0")}.json`, "json")
        .then(rows => {
          const expected = Math.min(shardSize, count - s * shardSize);
          if (!Array.isArray(rows) || rows.length !== expected || rows.some(r =>
            !r || typeof r.text !== "string" || typeof r.speaker !== "string" || typeof r.time !== "string")) {
            throw new Error(`Text shard ${s} does not match this dataset`);
          }
          return rows;
        }).catch(error => { shards.delete(s); throw error; });
      shards.set(s, request);
    }
    return shards.get(s);
  };
  const textOf = async i => {
    if (!Number.isInteger(i) || i < 0 || i >= count) throw new RangeError("Invalid turn index");
    return (await shardOf(Math.floor(i / shardSize)))[i % shardSize];
  };
  return { shardOf, textOf };
}

export function validateDataset(meta, stars, colors, edges, echoIndex, echoSim, context, contextKinds) {
  const check = (ok, message) => { if (!ok) throw new Error(message); };
  check(Number.isInteger(meta.count) && meta.count > 0 && Number.isInteger(meta.shard) && meta.shard > 0 &&
    Number.isFinite(meta.days) && meta.days >= 0 && Array.isArray(meta.speakers) && meta.speakers.length > 0 &&
    typeof meta.first === "string" && typeof meta.last === "string" &&
    Number.isFinite(Date.parse(meta.first.replace(" ", "T") + "Z")) &&
    meta.speakers.every(s => s && typeof s.name === "string" && Array.isArray(s.color) && s.color.length === 3 && s.color.every(Number.isFinite)),
    "Invalid dataset metadata");
  check(stars.byteLength === meta.count * 16 && colors.byteLength === meta.count * 4,
    "Star and color files do not match the dataset count");
  check(new Float32Array(stars).every(Number.isFinite), "Star file contains non-finite coordinates");
  const rgb = new Uint8Array(colors);
  for (let i = 0; i < meta.count; i++) check(rgb[i * 4 + 3] < meta.speakers.length, "Color file contains an invalid speaker index");
  const pairs = (buffer, name) => {
    check(buffer.byteLength % 8 === 0, `Invalid ${name} file`);
    check(new Uint32Array(buffer).every(i => i < meta.count), `${name} contains an invalid turn index`);
  };
  pairs(edges, "edges");
  check(Boolean(echoIndex) === Boolean(echoSim), "Echo index and similarity files must be present together");
  if (echoIndex) {
    check(echoIndex.byteLength % (meta.count * 4) === 0 && echoIndex.byteLength > 0 &&
      echoSim.byteLength * 4 === echoIndex.byteLength, "Echo files do not match the dataset count");
    check(new Uint32Array(echoIndex).every(i => i < meta.count), "Echo index contains an invalid turn index");
  }
  check(Boolean(context) === Boolean(contextKinds), "Context and context-kind files must be present together");
  if (context) {
    pairs(context, "context");
    check(contextKinds.byteLength === context.byteLength / 8 && new Uint8Array(contextKinds).every(k => k <= 1),
      "Context kinds do not match the context graph");
  }
}
