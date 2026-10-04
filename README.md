# Village Sky

**Live: https://asavs.github.io/village-sky/** (about 40 MB on first load; hosted with the AI Village
organisers' permission)

Every message ever posted in the [AI Village](https://theaidigest.org/village) as a star: 183,483 turns from 47
speakers, April 2025 to September 2026. Click one and its connections light up in place: what it read, who
read it, and where its idea echoed. Built for the AI Swarm Dynamics Hackathon (AI Village × Grove Research,
October 2026), whose brief was the tools investigators wished they had during the
[OpenAI / Hugging Face incident](https://metr.org/hugging-face-incident-report-aug-2026.pdf): about 1,200
agents on an unsanctioned message board, and a team tracing by hand which agents wrote and read which messages.

![The sky](docs/sky.jpg)

## Reading the sky

- **x, y: meaning.** A 2D map ([UMAP](https://umap-learn.readthedocs.io)) of each message's embedding
  (Qwen3-Embedding-0.6B). Messages that say similar things sit together, so a topic that persists over time
  forms a filament.
- **Depth: time.** The newest message is nearest. Depth is `log(1 + age / τ)`: the last weeks spread out,
  the first months become the far haze. The light from distant stars is old, as in the real sky.
- **Colour: model family.** Claude amber, GPT green, Gemini blue, o-series cyan, Grok red, DeepSeek violet,
  Kimi pink, GLM yellow, humans white.
- **Size: how many later turns had it in context.**

## Opening a turn

![A turn open in the sky](docs/turn.jpg)

Click a star. The camera turns to it, its text opens in a card beside it in its model's colour, and the
rest of the sky dims to a grey haze. What stays lit:

- **What it read:** its latest direct inputs (warm threads in) and its own previous message.
- **Who read it:** its first readers (cool threads out).
- **Its echoes:** the messages most similar to it in meaning, anywhere in time. Thread width and brightness
  show how strongly the idea could have been carried (see below).
- **Its light cone:** everything it could have influenced, or been influenced by, glowing faintly behind.

Bright streaks flow along the context threads: warm light into the turn, cool light toward its readers.
Its own sequence has a gentler flow toward later turns. Echo and search threads stay still; their
similarity does not imply transmission. The flow depicts direction in the inferred graph, not proven
information transfer, and is disabled when the system requests reduced motion.

Press **Z** to zoom into its moment: time re-centres on the turn, scaled to its conversation, and you see it
from the side, with what it read on the left and who read it on the right.
Arrow navigation keeps that time scale and camera framing while moving to the next turn. The card stays
open and shared connection labels remain visible; clicking a turn fits its conversation again.

![Zoomed into a moment](docs/moment.jpg)

| key | |
|---|---|
| click | open a turn (or a label) |
| ← → | step through search matches, or the speaker's turns; Shift ← → always follows the speaker |
| ↑ ↓ | open its latest input or first reader |
| Tab / hover | step through connection labels and show a text preview |
| Enter | go to that connection |
| Backspace | go back along your path |
| Z | zoom into its moment; F flips the zoomed view to look down the time axis |
| `[` `]` | dim the rest of the sky (overview: stretch recent time) |
| I | details: cone counts, echoes grouped by contact |
| Esc | back to the whole sky |
| drag, wheel, right-drag | look, fly through time, pan |
| A / H / L | autopilot (starts after 45 s idle) / hide the text / all edges |
| / / M | search messages and speakers / find messages similar to the open turn |

## What a connection means, and what it doesn't

The village is a shared chat. An agent does not see every message; when it next acts, the room's new messages
land in its context. The tool builds three kinds of edge from that, with different strength of evidence:

| edge | built from | strength |
|---|---|---|
| **read** | everything new in the room since that agent last spoke there (up to 50) | inferred from timing; prompts are not in the chat export |
| **memory** | the agent's own previous message, in any room | assumes it remembers what it said and read |
| **echo** | nearest neighbours by embedding similarity (≥ 0.80) | similarity of meaning only |

Two distances between any pair of turns, computed exactly by a breadth-first search that prunes by time:

- **hops:** reads and memory each cost 1. Short hops = the idea was fresh in context.
- **hand-offs:** only reads cost 1; an agent's own memory is free (a 0-1 BFS). This assumes perfect memory.

Real agents sit between the two. So the tool claims **"could have"**, never "did". An echo with **no path
in the inferred graph** has no known route through the modelled chat context. That does not prove
independent discovery: the export omits other read channels, and context edges are inferred. The UI calls
this group "no known path". If the full context files are absent, it reports reachability only in the
smaller drawn graph.

## Findings so far

Preliminary, from this dataset.

1. **Similar messages sit closer in the contact graph than chance.** Earlier echoes at 0.85–0.95 similarity
   lie within 3 hops 28–54% of the time, against 12% for a random earlier message at the same time gap: 2 to
   4 times as often, rising with similarity. (1,500 sampled turns; hops over the drawn edges.)
2. **With perfect memory, the village is a small world.** 70% of random earlier messages are one hand-off
   away. Contact alone cannot establish transmission in a shared room; meaning and short fresh-context
   distance together can suggest it; absence of a graph path alone does not prove independence.
3. **Phrases and conventions cross models, but contact does not explain the spread.** `tools/memes.py`
   clusters sentences into 3,225 families (cosine ≥ 0.88, ≥ 5 messages from ≥ 3 agents), covering 37% of
   messages. "I'll continue monitoring silently to avoid chat clutter" has 218 messages from 10 speakers
   across Claude, GPT, Gemini, DeepSeek and o3, with 7 jumps between model families. But a family's carriers
   have a short contact path (≤ 3 hops) to an earlier carrier no more often than time-matched random
   messages: the median lift is 1.01 (318 families above 1.2, 285 below 0.9; "monitoring silently" 0.94).
   The widest families are shared task state (milestone counts, charity totals, countdowns), where agents
   would converge on the same words without copying. Finding 1 is pair-level; at the family level, with a
   null that makes the same selection, the effect mostly disappears.
4. **So the chat cannot settle copying vs co-presence; other records can.** Nearly every family has a single
   root, because in a shared room everyone has a path to everyone. The dataset also holds direct evidence of
   what entered an agent's context: about 10k `SEARCH_HISTORY` answers quote earlier messages verbatim with
   timestamps, and `agent_memories` keeps full snapshots of what each agent wrote down. Those, not timing,
   are where transmission claims have to come from.

## Run it

Needs access to the dataset (gated: request it on the
[dataset page](https://huggingface.co/datasets/aidigestorg/ai-village), then `huggingface-cli login`), Python
3.11+, and a GPU for the embedding step (it runs on CPU, slowly).

```sh
pip install -r requirements.txt
huggingface-cli download aidigestorg/ai-village --repo-type dataset \
    --include chat_messages.jsonl.gz agents.jsonl.gz chat_rooms.jsonl.gz
python tools/embed.py       # message embeddings, ~1 h on an RTX 3070, resumable
python tools/layout.py      # the 2D map of meaning, ~2 min
python tools/export.py      # stars, edges, context graph, text shards, ~10 s
python tools/neighbors.py   # echoes: 24 nearest per message, ~40 s on a GPU
python -m http.server 8731  # then open http://localhost:8731
```

Everything runs locally and offline (three.js is vendored), which suits investigations on private data.
`data/` is not in the repository: it holds the dataset's text, and the dataset's terms ask for access
through its gate.

| file | |
|---|---|
| `index.html`, `sky.js` | the page: one three.js scene; stars, threads, cards and labels are all drawn on the GPU |
| `tools/village.py` | the dataset, and the one message order every file in `data/` follows |
| `tools/embed.py`, `tools/layout.py` | embeddings and the semantic map |
| `tools/export.py` | stars, drawn edges, the full context graph, text shards |
| `tools/neighbors.py` | the echo index |

Embedding parts now carry an input manifest. Resuming requires the same message IDs, text, order, and
encoding settings. If inputs change, or old parts have no manifest, use a new `--out` directory to rebuild;
existing parts are preserved. Only the parts expected for the current run are merged.

Run the focused regression checks without downloading the dataset or model:

```sh
npm test
npm run check
python -m unittest discover -s tests -p 'test_*.py'
```

## Next

The chat export is what everyone sees; the dataset also holds what each agent thought and kept.

- **`agent_memories`** (246k memories agents wrote for themselves): replace the memory assumption with
  evidence, by checking whether an idea is in an agent's written memory when it acts.
- **`events`**, `SEARCH_HISTORY` in particular: agents querying the village's past, a read channel the chat
  edges miss.
- **`computer_use_turns`** (2.5M turns with raw model output and reasoning): stars for every action, not only
  every message, and reasoning-level echoes.

## Data

AI Digest, "AI Village dataset", 2026. https://theaidigest.org/village. Used under its research terms:
research and analysis only, no training on it, no re-identification, cite AI Digest / AI Village.
