"""Transformer block definitions for PIXEL."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from configs.base import ModelConfig
from models.attention import GroupedQueryAttention
from models.moe import MoELayer
from models.norms import RMSNorm


class FeedForward(nn.Module):
    """Standard SwiGLU feed-forward block."""

    def __init__(self, config: ModelConfig):
        """Create the dense feed-forward module."""
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply a dense SwiGLU feed-forward step."""
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional MoE."""

    def __init__(self, config: ModelConfig, layer_index: int):
        """Create one transformer block."""
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_moe = config.use_moe and (layer_index + 1) % config.moe.expert_interval == 0
        self.feed_forward = MoELayer(config) if self.use_moe else FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Run one transformer block and return hidden states, cache, and aux loss."""
        attn_output, present = self.attention(self.norm1(hidden_states), cos, sin, past_key_value)
        hidden_states = hidden_states + attn_output
        aux_loss = hidden_states.new_zeros(())
        ff_input = self.norm2(hidden_states)
        if self.use_moe:
            ff_output, aux_loss = self.feed_forward(ff_input)
        else:
            ff_output = self.feed_forward(ff_input)
        hidden_states = hidden_states + ff_output
        return hidden_states, present, aux_loss
