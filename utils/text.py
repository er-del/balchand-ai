"""Text formatting helpers for PIXEL."""

from __future__ import annotations


def truncate_text(text: str, limit: int = 120) -> str:
    """Truncate a string for compact logs."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def clean_prompt(text: str) -> str:
    """Normalize prompt whitespace without changing intent."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()
