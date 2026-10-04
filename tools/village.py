"""The AI Village dataset, as the other tools read it.

The snapshot is the Hugging Face cache of aidigestorg/ai-village (gated: request access on its page, then
`huggingface-cli login`). messages() defines the one selection and order every file in data/ depends on:
non-empty chat messages, sorted by time. Row i of the embeddings, the semantic map, and stars.bin is the
same message.
"""
import glob, gzip, json, os, sys
from functools import lru_cache

here = os.path.dirname(os.path.abspath(__file__))
data = os.path.normpath(os.path.join(here, "..", "data"))


@lru_cache(maxsize=1)
def snapshot():
    """The newest cached snapshot that has the chat messages."""
    paths = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--aidigestorg--ai-village/snapshots/*"))
    paths = [p for p in paths if os.path.exists(os.path.join(p, "chat_messages.jsonl.gz"))]
    if not paths:
        sys.exit("AI Village dataset not found. Request access at https://huggingface.co/datasets/aidigestorg/ai-village, "
                 "log in, then: huggingface-cli download aidigestorg/ai-village --repo-type dataset "
                 "--include 'chat_messages.jsonl.gz' 'agents.jsonl.gz' 'chat_rooms.jsonl.gz'")
    return max(paths, key=os.path.getmtime)


def rows(name):
    with gzip.open(os.path.join(snapshot(), name), "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def messages():
    """[(meta, text)]: every non-empty chat message, by time. meta has id, time, speaker, room."""
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
    return items
