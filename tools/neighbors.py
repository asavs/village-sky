"""Nearest neighbours in meaning for every message: the echoes.

Exact cosine kNN over the message embeddings (data/embeddings, from embed.py) (Qwen3, 256 dims), on the GPU in blocks. Writes:

  data/echo_index.bin  uint32, K per star, nearest first (self and empty rows excluded)
  data/echo_sim.bin    uint8,  K per star, cosine similarity * 255

    python tools/neighbors.py [K]
"""
import os, sys, time
import numpy as np, torch

from village import data as out

emb = os.path.join(out, "embeddings")
K = int(sys.argv[1]) if len(sys.argv) > 1 else 24

t = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.from_numpy(np.load(os.path.join(emb, "messages.f16.npy")).astype(np.float32)).to(device)
x = torch.nn.functional.normalize(x, dim=1).half()
n = x.shape[0]
index = np.empty((n, K), np.uint32)
sim = np.empty((n, K), np.uint8)
for a in range(0, n, 4096):
    s = x[a:a + 4096] @ x.T
    s[torch.arange(s.shape[0]), torch.arange(a, a + s.shape[0])] = -1    # not your own echo
    v, i = s.float().topk(K, dim=1)
    index[a:a + 4096] = i.cpu().numpy()
    sim[a:a + 4096] = (v.clamp(0, 1) * 255).round().byte().cpu().numpy()
index.tofile(os.path.join(out, "echo_index.bin"))
sim.tofile(os.path.join(out, "echo_sim.bin"))
q = sim[:, 0] / 255
print(f"{n} stars, K={K}, {device}, {time.time() - t:.0f}s; nearest similarity p10/50/90/99:",
      np.percentile(q, [10, 50, 90, 99]).round(3))
