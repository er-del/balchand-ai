"""Rotary positional embedding helpers for PIXEL."""

from __future__ import annotations

import torch


def build_rope_cache(seq_len: int, head_dim: int, base: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine tables for RoPE."""
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate alternating hidden dimensions."""
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(start_dim=-2)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    cos = cos.unsqueeze(0).unsqueeze(0).repeat_interleave(2, dim=-1)
    sin = sin.unsqueeze(0).unsqueeze(0).repeat_interleave(2, dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
