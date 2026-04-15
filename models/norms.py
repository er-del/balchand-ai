"""Normalization layers used by PIXEL."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Apply RMS normalization without centering."""

    def __init__(self, hidden_size: int, eps: float = 1.0e-5):
        """Create one RMSNorm layer."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension and scale it."""
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        return normalized * self.weight
