"""Dataset normalization and token caching for PIXEL."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from tokenizer.manager import PixelTokenizer
from utils.files import ensure_dir


def _read_lines(path: Path) -> list[str]:
    """Read normalized text samples from a supported file format."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".jsonl":
        lines: list[str] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            text = payload.get("text", "")
            if isinstance(text, str) and text.strip():
                lines.append(text.strip())
        return lines
    if suffix == ".parquet":
        table = pq.read_table(path)
        if "text" not in table.column_names:
            raise KeyError(f"Parquet file missing text column: {path}")
        return [str(item).strip() for item in table.column("text").to_pylist() if str(item).strip()]
    raise ValueError(f"Unsupported data file: {path}")


def normalize_corpus(paths: Iterable[str]) -> list[str]:
    """Read all supported data files into a single sample list."""
    samples: list[str] = []
    for item in paths:
        samples.extend(_read_lines(Path(item)))
    return samples


def cache_tokenized_corpus(samples: list[str], tokenizer: PixelTokenizer, cache_dir: str | Path, sequence_length: int) -> Path:
    """Cache tokenized samples to speed up repeated small runs."""
    ensure_dir(cache_dir)
    fingerprint = hashlib.sha256()
    for sample in samples:
        fingerprint.update(sample.encode("utf-8"))
        fingerprint.update(b"\0")
    fingerprint.update(str(tokenizer.vocab_size).encode("utf-8"))
    fingerprint.update(str(sequence_length).encode("utf-8"))
    cache_path = Path(cache_dir) / f"{fingerprint.hexdigest()[:16]}.pt"
    if cache_path.exists():
        return cache_path
    sequences: list[torch.Tensor] = []
    for sample in samples:
        token_ids = tokenizer.encode(sample, add_bos=True, add_eos=True)
        if len(token_ids) < 2:
            continue
        for start in range(0, max(len(token_ids) - 1, 1), sequence_length):
            window = token_ids[start : start + sequence_length + 1]
            if len(window) < 2:
                continue
            sequences.append(torch.tensor(window, dtype=torch.long))
    torch.save(sequences, cache_path)
    return cache_path


@dataclass(slots=True)
class TokenDatasetConfig:
    """Configure token dataset loading."""

    paths: tuple[str, ...]
    sequence_length: int
    cache_dir: str = "artifacts/cache"


class TokenDataset(Dataset[dict[str, torch.Tensor]]):
    """Provide fixed-length token windows for language modeling."""

    def __init__(self, config: TokenDatasetConfig, tokenizer: PixelTokenizer):
        """Read, tokenize, and cache samples for one dataset."""
        samples = normalize_corpus(config.paths)
        if not samples:
            raise ValueError("No training samples were found.")
        cache_path = cache_tokenized_corpus(samples, tokenizer, config.cache_dir, config.sequence_length)
        self.sequences: list[torch.Tensor] = torch.load(cache_path)

    def __len__(self) -> int:
        """Return the number of token windows."""
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one token window split into inputs and labels."""
        sequence = self.sequences[index]
        return {
            "input_ids": sequence[:-1],
            "labels": sequence[1:],
        }
