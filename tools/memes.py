"""Meme families and how they could have spread: phrases that recur across speakers, traced over the context graph.

A meme is a family of sentence spans that say nearly the same thing. Spans come from turnviz (loom/data/embeddings:
spans.f16.npy, Qwen3 256 dims, and spans.jsonl); messages, text and the context graph from export.py. Families are
collapsed to messages: a family's carriers are the distinct stars with a span in it.

For each carrier after the first, could_have_come_from is the earlier carrier it is closest to over the context graph:
fewest hand-offs (reads cost 1, the speaker's own memory 0), then fewest fresh hops (every edge 1), then most recent.
Both are exact shortest paths in the DAG, which only runs forward in time. A carrier with no path from any earlier
carrier cannot have received the meme through the chat: it is a root, found independently (or from outside the chat).
Everything else is a path that existed, not a transmission that happened. Lift compares how often cross-speaker
parents sit within 3 fresh hops with a time-matched null, which is the part of the signal contact alone cannot give.

Writes into data/:

  memes.json         {"params", "families": [...]}, sorted by spread = near_observed - near_expected
  meme_of_star.bin   int32 per star, the family holding most of the star's spans, or -1
  embeddings/memes.knn.npz  cached span kNN (delete it to recompute)

    python tools/memes.py [similarity] [min-chars]
"""
import glob, json, os, re, sys, time
from collections import Counter
import numpy as np, torch, numba
import igraph as ig, leidenalg as la
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from village import data as out

TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.88      # cosine for a span-span edge, after removing names
MINCHARS = int(sys.argv[2]) if len(sys.argv) > 2 else 30     # span length with speaker names removed
K = 16                  # neighbours per span; edges must be mutual
MAXSPANS = 2000         # components bigger than this are split by Leiden
NEAR = 3                # "close" in fresh hops, for lift
NULLDRAWS = 10
SPANS = r"C:\Users\asas\Research\ai-village\turnviz\loom\data\embeddings"
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(*@#-])|\n+")     # as turnviz split messages into spans
clock = time.time()
def stage(name):
    global clock
    print(f"{name}: {time.time() - clock:.0f}s", flush=True)
    clock = time.time()

# --- stars, speakers, text, spans
emb = os.path.join(out, "embeddings")
star_of = {json.loads(l)["id"]: i for i, l in enumerate(open(os.path.join(emb, "messages.jsonl"), encoding="utf-8"))}
n = len(star_of)
names = [s["name"] for s in json.load(open(os.path.join(out, "meta.json")))["speakers"]]
speaker = np.fromfile(os.path.join(out, "colors.bin"), np.uint8).reshape(-1, 4)[:, 3].astype(np.int32)
days = np.fromfile(os.path.join(out, "stars.bin"), np.float32).reshape(-1, 4)[:, 2].astype(np.float64)
human = names.index("human")
text = []
for f in sorted(glob.glob(os.path.join(out, "text", "*.json"))):
    text += [m["text"] for m in json.load(open(f, encoding="utf-8"))]
span_msg, span_idx = [], []
for l in open(os.path.join(SPANS, "spans.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    span_msg.append(star_of[r["message"]]); span_idx.append(r["index"])
span_msg = np.array(span_msg, np.int32)
span_text, last, pieces = [], -1, []
for m, k in zip(span_msg, span_idx):
    if m != last:
        pieces, last = [p.strip() for p in SPLIT.split(text[m])], m
    span_text.append(pieces[k])
stage(f"spans {len(span_text)}")

# Spans cluster by who they mention ("@DeepSeek-V3.2 thanks!") before what they say, so names come out twice: from
# the length test, and from the embedding as the directions that separate spans naming each model from the rest.
tokens = sorted({nm.lower() for nm in names if nm != "human"} |
                {"claude", "opus", "sonnet", "haiku", "fable", "gemini", "grok", "deepseek", "kimi", "glm", "gpt", "o3", "o1"}, key=len, reverse=True)
lower = [s.lower() for s in span_text]
mentions = {t: np.array([t in s for s in lower]) for t in tokens}
short = {"o3", "o1"}   # too short for substring tests; only used as directions when they stand alone
stripped = np.array([len(s) for s in span_text])
for i in np.where(np.any([v for t, v in mentions.items() if t not in short], axis=0))[0]:
    s = lower[i]
    for t in tokens:
        s = s.replace(t, "")
    stripped[i] = len(s.strip(" @,:-*"))
eligible = np.where(stripped >= MINCHARS)[0]
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.from_numpy(np.load(os.path.join(SPANS, "spans.f16.npy"))).to(device).float()
mu = x.mean(0)
dirs = [x[torch.from_numpy(v).to(device)].mean(0) - mu for t, v in mentions.items()
        if v.sum() >= 100 and t not in short]
Q, _ = torch.linalg.qr(torch.stack(dirs, 1))
x = torch.nn.functional.normalize(x - (x @ Q) @ Q.T, dim=1).half()
stage(f"names: {Q.shape[1]} directions removed, {len(eligible)} spans of {MINCHARS}+ chars")

# --- exact kNN of eligible spans on the GPU
cache = os.path.join(emb, "memes.knn.npz")
if os.path.exists(cache) and np.load(cache)["eligible"].shape == eligible.shape and (np.load(cache)["eligible"] == eligible).all():
    c = np.load(cache); I, S = c["I"], c["S"]
else:
    xe = x[torch.from_numpy(eligible).to(device)]
    m = len(eligible)
    I = np.empty((m, K), np.int32); S = np.empty((m, K), np.float16)
    for a in range(0, m, 1024):
        s = xe[a:a + 1024] @ xe.T
        s[torch.arange(s.shape[0]), torch.arange(a, a + s.shape[0])] = -1
        v, i = s.topk(K, dim=1)
        I[a:a + 1024] = i.cpu().numpy(); S[a:a + 1024] = v.cpu().numpy()
    np.savez(cache, eligible=eligible, I=I, S=S)
    del xe
stage("kNN")

# --- families. Mutual kNN edges above TAU between spans of different messages; connected components keep tight
# paraphrase sets whole, and the few components that chain into blobs are split by Leiden (modularity, weighted)
# rather than raising TAU for everyone.
m = len(eligible)
r = np.repeat(np.arange(m), K); c = I.ravel(); s = S.ravel().astype(np.float32)
keep = (s >= TAU) & (span_msg[eligible[r]] != span_msg[eligible[c]])
A = coo_matrix((s[keep], (r[keep], c[keep])), shape=(m, m)).tocsr()
A = A.minimum(A.T)                                   # mutual: both ends must list each other
_, label = connected_components(A, directed=False)
label = label.astype(np.int64)
sizes = np.bincount(label)
nextlab = label.max() + 1
todo = [np.where(label == q)[0] for q in np.where(sizes > MAXSPANS)[0]]
for depth in range(4):
    later = []
    for nodes in todo:
        sub = A[nodes][:, nodes].tocoo()
        up = sub.row < sub.col
        g = ig.Graph(n=len(nodes), edges=np.stack([sub.row[up], sub.col[up]], 1).tolist(), edge_attrs={"w": sub.data[up]})
        part = np.array(la.find_partition(g, la.ModularityVertexPartition, weights="w", seed=depth).membership)
        if part.max() == 0:
            continue
        label[nodes] = nextlab + part
        nextlab += part.max() + 1
        later += [nodes[part == q] for q in range(part.max() + 1) if (part == q).sum() > MAXSPANS]
    todo = later
stage(f"families: {(sizes > 1).sum()} components, {(sizes > MAXSPANS).sum()} split")

# collapse to messages; strength = spans of the star in the family
fam_of_span = label
star_of_span = span_msg[eligible]
order = np.lexsort((star_of_span, fam_of_span))
fs, ss = fam_of_span[order], star_of_span[order]
bounds = np.flatnonzero(np.diff(fs)) + 1
families = []
for grp in np.split(np.arange(len(fs)), bounds):
    stars, counts = np.unique(ss[grp], return_counts=True)
    if len(stars) < 5:
        continue
    spk = set(speaker[stars].tolist()) - {human}
    if len(spk) < 3:
        continue
    o = np.argsort(stars, kind="stable")
    families.append({"spans": order[grp], "carriers": stars[o], "strength": counts[o]})
stage(f"kept {len(families)} families")

# --- the context graph, as in-edges per star (sources are always earlier)
pairs = np.fromfile(os.path.join(out, "context.bin"), np.uint32).reshape(-1, 2).astype(np.int64)
kinds = np.fromfile(os.path.join(out, "context_kinds.bin"), np.uint8)
o = np.lexsort((pairs[:, 0], pairs[:, 1]))
src = pairs[o, 0].astype(np.int32); cost = kinds[o].astype(np.int16)
ptr = np.zeros(n + 1, np.int64); np.add.at(ptr, pairs[:, 1] + 1, 1); ptr = np.cumsum(ptr)
INF = 32000

@numba.njit(parallel=True, cache=True)
def distances(carriers, ptr, src, cost):
    """hand-offs and fresh hops from every carrier to every later one: one forward pass over the DAG per source."""
    k = len(carriers)
    H = np.full((k, k), INF, np.int16); P = np.full((k, k), INF, np.int16)
    end = carriers[-1]
    for j in numba.prange(k - 1):
        a = carriers[j]
        hand = np.full(end - a + 1, INF, np.int16); hop = np.full(end - a + 1, INF, np.int16)
        hand[0] = 0; hop[0] = 0
        nxt = j + 1
        for v in range(a + 1, end + 1):
            h = INF; p = INF
            for e in range(ptr[v], ptr[v + 1]):
                u = src[e]
                if u < a:
                    continue
                x = hand[u - a] + cost[e]
                if x < h: h = x
                y = hop[u - a] + 1
                if y < p: p = y
            hand[v - a] = h; hop[v - a] = p
            while nxt < k and carriers[nxt] == v:
                H[j, nxt] = h; P[j, nxt] = p
                nxt += 1
    return H, P

@numba.njit(cache=True)
def near_back(c, candidates, depth, ptr, src, mark, stamp, queue):
    """which candidates reach c within `depth` fresh hops (backward BFS from c)."""
    queue[0] = c; mark[c] = stamp
    lo, hi = 0, 1
    for _ in range(depth):
        end = hi
        for q in range(lo, end):
            v = queue[q]
            for e in range(ptr[v], ptr[v + 1]):
                u = src[e]
                if mark[u] != stamp:
                    mark[u] = stamp; queue[hi] = u; hi += 1
        lo = end
    res = np.zeros(len(candidates), np.bool_)
    for i in range(len(candidates)):
        res[i] = mark[candidates[i]] == stamp and candidates[i] != c
    return res

def model(name):
    low = name.lower()
    if name == "human": return "human"
    for key in ("claude", "gpt", "gemini", "grok", "deepseek", "kimi", "glm"):
        if key in low: return key
    return "o-series" if re.match(r"o\d", low) else "other"
model_of = np.array([model(nm) for nm in names])

rng = np.random.default_rng(0)
mark = np.zeros(n, np.int32); stamp = 0; queue = np.empty(n, np.int64)
xs = x.float()
for f in families:
    car = f["carriers"]
    k = len(car)
    H, P = distances(car.astype(np.int64), ptr, src, cost)
    parent = np.full(k, -1, np.int64); hand = np.full(k, -1); hops = np.full(k, -1)
    for b in range(1, k):
        h, p = H[:b, b].astype(np.int64), P[:b, b].astype(np.int64)
        if h.min() >= INF:
            continue
        j = np.lexsort((-np.arange(b), p, h))[0]        # fewest hand-offs, then hops, then most recent
        parent[b], hand[b], hops[b] = j, h[j], p[j]
    sp = speaker[car]
    # Lift, two ways, both on other speakers only (an agent's own earlier message is one memory step away anyway).
    # pair_lift: is the chosen parent within NEAR fresh hops, vs one random message of another speaker at the same
    # time gap. It flatters the family: the parent was picked as the closest of many earlier carriers.
    links = [b for b in range(k) if parent[b] >= 0 and sp[parent[b]] != sp[b]]
    obs = sum(hops[b] <= NEAR for b in links)
    null = draws = 0
    for b in links:
        c, gap = car[b], days[car[b]] - days[car[parent[b]]]
        at = np.searchsorted(days, days[c] - gap)
        pool = np.arange(max(0, at - 60), min(c, at + 60))
        pool = pool[speaker[pool] != sp[b]]
        if len(pool) == 0:
            continue
        pick = rng.choice(pool, min(NULLDRAWS, len(pool)), replace=False)
        stamp += 1
        null += near_back(c, pick, NEAR, ptr, src, mark, stamp, queue).sum(); draws += len(pick)
    pair_lift = ((obs + 1) / (len(links) + 2)) / ((null + 1) / (draws + 2)) if links else None
    # lift: is ANY earlier carrier of another speaker within NEAR fresh hops, vs the same question with each of those
    # carriers swapped for a random other-speaker message near its time. Same selection on both sides.
    near = expect = trials = 0
    for b in range(1, k):
        E = np.where(sp[:b] != sp[b])[0]
        if len(E) == 0:
            continue
        c = car[b]
        trials += 1
        near += P[E, b].min() <= NEAR
        cand = np.clip(car[E][None, :, None] + rng.integers(-60, 61, (NULLDRAWS, len(E), 4)), 0, c - 1)
        ok = speaker[cand] != sp[b]                     # first of four tries that is another speaker
        pick = np.take_along_axis(cand, ok.argmax(2)[..., None], 2)[..., 0]
        stamp += 1
        hit = near_back(c, pick.ravel(), NEAR, ptr, src, mark, stamp, queue).reshape(pick.shape) & ok.any(2)
        expect += hit.any(1).mean()
    # one pseudo-count each side shrinks small families toward no lift
    f["lift"] = (near + 1) / (expect + 1) if trials else None
    f.update(parent=parent, hand=hand, hops=hops, links=len(links), pair_lift=pair_lift, trials=trials,
             near=int(near), expect=float(expect))
    f["children"] = np.bincount(parent[parent >= 0], minlength=k)
    f["cross_speaker"] = int(sum(1 for b in range(k) if parent[b] >= 0 and sp[parent[b]] != sp[b]))
    f["cross_model"] = int(sum(1 for b in range(k) if parent[b] >= 0 and model_of[sp[parent[b]]] != model_of[sp[b]]
                               and "human" not in (model_of[sp[parent[b]]], model_of[sp[b]])))
    # spread: carriers in contact with another speaker's earlier carrier beyond what timing alone predicts. Lift
    # alone favours tiny families; speakers x lift crowns generic thanks, which reach everyone and gain nothing.
    f["score"] = near - expect
    # representative: the span nearest the family centroid
    v = xs[torch.from_numpy(eligible[f["spans"]]).to(device)]
    f["phrase"] = span_text[eligible[f["spans"]][int((v @ v.mean(0)).argmax())]][:140]
stage("distances and lift")

families.sort(key=lambda f: -f["score"])
meme_of_star = np.full(n, -1, np.int32)
best = np.zeros(n, np.int32)
for fid, f in enumerate(families):           # strongest = most spans; ties go to the higher-spread family
    car, st = f["carriers"], f["strength"]
    win = st > best[car]
    meme_of_star[car[win]] = fid; best[car[win]] = st[win]
meme_of_star.tofile(os.path.join(out, "meme_of_star.bin"))
day0 = lambda d: int(np.floor(d))
rows = []
for fid, f in enumerate(families):
    car, sp = f["carriers"], speaker[f["carriers"]]
    adopt = Counter(day0(days[c]) for c in car)
    roots = int((f["parent"] < 0).sum())
    rows.append({
        "id": fid, "phrase": f["phrase"], "carriers": car.tolist(),
        "could_have_come_from": [int(car[p]) if p >= 0 else -1 for p in f["parent"]],
        "handoffs": f["hand"].tolist(), "hops": f["hops"].tolist(),
        "roots": roots, "R": round(float(f["children"].mean()), 3),
        "max_children": int(f["children"].max()),
        "speakers": sorted({names[s] for s in sp}), "models": sorted({model_of[s] for s in sp}),
        "cross_speaker": f["cross_speaker"], "cross_model_jumps": f["cross_model"],
        "near_trials": f["trials"], "near_observed": f["near"], "near_expected": round(f["expect"], 2),
        "lift": None if f["lift"] is None else round(f["lift"], 3),
        "pair_lift": None if f["pair_lift"] is None else round(f["pair_lift"], 3), "spread": round(f["score"], 2),
        "origin": int(car[0]), "first_day": round(float(days[car[0]]), 3), "last_day": round(float(days[car[-1]]), 3),
        "adoption": sorted(adopt.items()),
    })
json.dump({"params": {"similarity": TAU, "min_chars": MINCHARS, "k": K, "mutual": True, "max_spans": MAXSPANS,
                      "near_hops": NEAR, "null_draws": NULLDRAWS, "lift": "(near_observed+1)/(near_expected+1): carriers with an other-speaker earlier carrier within near_hops, vs time-matched swaps",
                      "pair_lift": "parent within near_hops vs one time-gap-matched random message; flattered by choosing the closest parent",
                      "could_have_come_from": "star index of the earlier carrier at fewest hand-offs, then hops, then most recent; -1 = no path, a root",
                      "first": json.load(open(os.path.join(out, "meta.json")))["first"]},
           "families": rows}, open(os.path.join(out, "memes.json"), "w", encoding="utf-8"), ensure_ascii=False)
stage("write")

covered = (meme_of_star >= 0).sum()
print(f"{len(rows)} families, {covered} stars carry one ({covered / n:.1%}), roots {sum(r['roots'] for r in rows)}")
for r in rows[:15]:
    print(f"{r['spread']:6.1f} lift {r['lift'] or 0:4.2f} pair {r['pair_lift'] or 0:5.2f} spk {len(r['speakers']):2d} car {len(r['carriers']):5d} roots {r['roots']:3d} "
          f"xmodel {r['cross_model_jumps']:3d} day {r['first_day']:5.0f}  {r['phrase'][:80]}")
