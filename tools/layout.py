"""The semantic map: UMAP of the message embeddings to 2D (cosine, 30 neighbours, min_dist 0.15).
Messages that mean similar things land near each other; the sky uses this as x, y. About 2 minutes.

    python tools/layout.py
"""
import os, time
import numpy as np, umap
from village import data

t = time.time()
path = os.path.join(data, "embeddings")
x = np.load(os.path.join(path, "messages.f16.npy")).astype(np.float32)
x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
y = umap.UMAP(n_neighbors=30, min_dist=0.15, metric="cosine", low_memory=True, random_state=None).fit_transform(x)
np.save(os.path.join(path, "messages.umap2.f32.npy"), y.astype(np.float32))
print("map", y.shape, f"{time.time() - t:.0f}s")
