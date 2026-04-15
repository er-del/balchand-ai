"""Tests for PIXEL tokenizer management."""

from pathlib import Path

import pytest

import tokenizer.manager as manager
from tokenizer.manager import ensure_tokenizer


def test_ensure_tokenizer_trains_and_roundtrips(tmp_path: Path) -> None:
    corpus = tmp_path / "demo.txt"
    corpus.write_text("hello world\nhello pixel\n", encoding="utf-8")
    tokenizer = ensure_tokenizer(model_prefix=str(tmp_path / "tok"), data_paths=[str(corpus)], vocab_size=512)
    assert Path(tokenizer.model_path).exists()
    text = "hello pixel"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_load_binary_model_without_sentencepiece_falls_back(monkeypatch, tmp_path: Path) -> None:
    """Binary tokenizer files should not crash when sentencepiece is unavailable."""
    model_path = tmp_path / "tok.model"
    model_path.write_bytes(b"\x80\x03binary-tokenizer-content")
    monkeypatch.setattr(manager, "spm", None)
    with pytest.warns(RuntimeWarning, match="sentencepiece is unavailable"):
        tokenizer = manager.PixelTokenizer.load(model_path)
    text = "fallback is robust"
    assert tokenizer.decode(tokenizer.encode(text)) == text
