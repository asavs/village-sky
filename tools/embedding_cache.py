"""Bind resumable embedding parts to their input text and encoding settings."""
import hashlib
import json
from pathlib import Path


def prepare_parts(directory, items, settings):
    digest = hashlib.sha256()
    for meta, text in items:
        digest.update(json.dumps([meta["id"], text], ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    expected = {"input_sha256": digest.hexdigest(), "count": len(items), "settings": settings}
    parts = Path(directory)
    parts.mkdir(parents=True, exist_ok=True)
    manifest = parts / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != expected:
            raise ValueError("Embedding inputs or settings changed. Use a new --out directory to avoid mixing cached parts.")
    elif any(parts.glob("part-*.npy")):
        raise ValueError("Cached parts have no input manifest. Use a new --out directory to rebuild safely.")
    else:
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(expected, indent=2), encoding="utf-8")
        temporary.replace(manifest)
    return parts
