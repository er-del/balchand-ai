"""Bootstrap corpus generation for zero-argument PIXEL training."""

from __future__ import annotations

from pathlib import Path

from tokenizer.bootstrap_text import BOOTSTRAP_TEXTS


def ensure_bootstrap_corpus(path: str | Path = "data/bootstrap/demo_corpus.txt") -> Path:
    """Write a small local training corpus when one does not exist."""
    target = Path(path)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    repeated: list[str] = []
    for _ in range(32):
        repeated.extend(BOOTSTRAP_TEXTS)
    target.write_text("\n".join(repeated) + "\n", encoding="utf-8")
    return target
