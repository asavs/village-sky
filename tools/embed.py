"""Embeds every chat message: Qwen3-Embedding-0.6B, Matryoshka-sliced to 256 dims, unit length, float16.

Writes data/embeddings/part-NNNN.npy in parts of 5,000 rows, so a crash loses minutes and a rerun skips
finished parts; then merges them into data/embeddings/messages.f16.npy and messages.jsonl (one id per row).
No instruction prefix: message-to-message similarity is symmetric. Qwen3-Embedding pools the last token,
so padding goes on the left. About an hour on an RTX 3070 for 183k messages.

    python tools/embed.py [--batch 32] [--limit N]
"""
import argparse, glob, json, os, time
from village import data, messages

MODEL, DIMS, PART = "Qwen/Qwen3-Embedding-0.6B", 256, 5000
out = os.path.join(data, "embeddings")

p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=32)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--out", default=out)
args = p.parse_args()

import numpy as np, torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL, device=device, model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32},
                            tokenizer_kwargs={"padding_side": "left"})
model.max_seq_length = 2048
items = messages()[: args.limit or None]
parts = os.path.join(args.out, "parts")
os.makedirs(parts, exist_ok=True)
started = time.time()
for part, start in enumerate(range(0, len(items), PART)):
    path = os.path.join(parts, "part-%04d.npy" % part)
    if os.path.exists(path):
        continue
    chunk = items[start:start + PART]
    v = model.encode([t for _, t in chunk], batch_size=args.batch, convert_to_numpy=True, show_progress_bar=False)[:, :DIMS]
    v = (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)).astype(np.float16)
    np.save(path + ".tmp.npy", v)
    os.replace(path + ".tmp.npy", path)
    print(f"part {part}: {start + len(chunk)}/{len(items)}  {(time.time() - started) / 60:.1f} min", flush=True)

files = sorted(glob.glob(os.path.join(parts, "part-*.npy")))
np.save(os.path.join(args.out, "messages.f16.npy"), np.concatenate([np.load(f) for f in files]))
with open(os.path.join(args.out, "messages.jsonl"), "w", encoding="utf-8") as f:
    for meta, _ in items:
        f.write(json.dumps({"id": meta["id"]}) + "\n")
print("messages", len(items), "->", args.out)
