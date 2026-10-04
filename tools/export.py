"""Exports every AI Village chat message as a star.

Reads the Hugging Face snapshot and the semantic map from turnviz (messages.umap2.f32.npy, one row per
message in the same order as messages.jsonl). Writes into data/:

  stars.bin   per star, float32 x, y, days (since the first message), reads (how many later messages had it in context)
  colors.bin  per star, uint8 r, g, b, speaker index
  edges.bin   uint32 pairs (earlier, later): the speaker's previous message (chain) and the last few
              messages in the room since then (read)
  meta.json   speakers, colours, rooms, time range, counts
  text/NNNN.json  message text in shards of 2,000 stars, fetched when a star is opened

    python tools/export.py [turnviz-embeddings-dir] [reads-per-turn]
"""
import glob, gzip, json, os, struct, sys
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.normpath(os.path.join(here, "..", "data"))
emb = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "turnviz", "loom", "data", "embeddings")
recent = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SHARD = 2000

snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--aidigestorg--ai-village/snapshots/*"))[0]
def rows(name):
    with gzip.open(os.path.join(snap, name), "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

# same selection and order as turnviz's embed.py, so row i here is row i of the semantic map
agents = {r["id"]: r["name"] for r in rows("agents.jsonl.gz")}
rooms = {r["id"]: r["name"] for r in rows("chat_rooms.jsonl.gz")}
items = []
for r in rows("chat_messages.jsonl.gz"):
    text = (r.get("content") or "").strip()
    if not text:
        continue
    speaker = agents.get(r["agent_speaker_id"], "human") if r["speaker_type"] == "agent" else "human"
    items.append(({"id": r["id"], "time": r["created_at"][:19], "speaker": speaker, "room": rooms.get(r["room_id"], "?")}, text))
items.sort(key=lambda it: it[0]["time"])

ids = [json.loads(l)["id"] for l in open(os.path.join(emb, "messages.jsonl"), encoding="utf-8")]
assert ids == [m["id"] for m, _ in items], "message order differs from the semantic map"
xy = np.load(os.path.join(emb, "messages.umap2.f32.npy"))
xy = (xy - np.median(xy, axis=0)) / np.percentile(np.abs(xy - np.median(xy, axis=0)), 98)   # about -1..1
n = len(items)

# colour by model family, varied a little per model
FAMILIES = [("claude", (1.00, 0.62, 0.30)), ("fable", (1.00, 0.80, 0.45)), ("gpt", (0.35, 0.95, 0.65)),
            ("o1", (0.40, 0.90, 0.95)), ("o3", (0.40, 0.90, 0.95)), ("o4", (0.40, 0.90, 0.95)),
            ("gemini", (0.45, 0.65, 1.00)), ("grok", (1.00, 0.40, 0.45)), ("deepseek", (0.70, 0.50, 1.00)),
            ("kimi", (1.00, 0.50, 0.85)), ("glm", (0.95, 0.95, 0.40)), ("human", (1.00, 1.00, 1.00))]
speakers = sorted({m["speaker"] for m, _ in items})
def colour(name, k):
    low = name.lower()
    base = next((c for key, c in FAMILIES if key in low), (0.75, 0.70, 0.90))
    if name == "human":
        return base
    shift = ((k * 0.618) % 1.0 - 0.5) * 0.25
    return tuple(min(1.0, max(0.0, c + shift * (1 if i == k % 3 else -0.5))) for i, c in enumerate(base))
speaker_colour = {s: colour(s, k) for k, s in enumerate(speakers)}
speaker_index = {s: k for k, s in enumerate(speakers)}

# edges: chain (speaker's previous message) and reads (last few room messages since then)
from datetime import datetime
t0 = datetime.fromisoformat(items[0][0]["time"])
days = np.array([(datetime.fromisoformat(m["time"]) - t0).total_seconds() / 86400 for m, _ in items], np.float32)
edges = []
reads = np.zeros(n, np.float32)
last_by = {}           # (speaker, room) -> index
room_log = {}          # room -> list of indices
for i, (m, _) in enumerate(items):
    log = room_log.setdefault(m["room"], [])
    prev = last_by.get((m["speaker"], m["room"]))
    if prev is not None:
        edges.append((prev, i, 0))
    new = [j for j in log[-50:] if (prev is None or j > prev) and items[j][0]["speaker"] != m["speaker"]]
    for j in new:
        reads[j] += 1                  # influence counts the whole context, edges draw only the latest
    edges.extend((j, i, 1) for j in new[-recent:])
    last_by[(m["speaker"], m["room"])] = i
    log.append(i)

os.makedirs(os.path.join(out, "text"), exist_ok=True)
np.stack([xy[:, 0], xy[:, 1], days, reads], 1).astype(np.float32).tofile(os.path.join(out, "stars.bin"))
cols = np.array([[*(int(255 * c) for c in speaker_colour[m["speaker"]]), speaker_index[m["speaker"]]] for m, _ in items], np.uint8)
cols.tofile(os.path.join(out, "colors.bin"))
np.array([(a, b) for a, b, _ in edges], np.uint32).tofile(os.path.join(out, "edges.bin"))
np.array([k for _, _, k in edges], np.uint8).tofile(os.path.join(out, "edge_kinds.bin"))
for s in range(0, n, SHARD):
    shard = [{"id": m["id"], "time": m["time"], "speaker": m["speaker"], "room": m["room"], "text": t} for m, t in items[s:s + SHARD]]
    json.dump(shard, open(os.path.join(out, "text", "%04d.json" % (s // SHARD)), "w", encoding="utf-8"), ensure_ascii=False)
json.dump({"count": n, "edges": len(edges), "shard": SHARD, "first": items[0][0]["time"], "last": items[-1][0]["time"],
           "days": float(days[-1]), "speakers": [{"name": s, "color": speaker_colour[s]} for s in speakers]},
          open(os.path.join(out, "meta.json"), "w"), indent=1)
print("stars", n, "edges", len(edges), "speakers", len(speakers), "days %.1f" % days[-1], "max reads", int(reads.max()))
