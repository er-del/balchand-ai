"""Grouped-query attention for PIXEL."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from configs.base import ModelConfig
from models.rope import apply_rope


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat key or value heads to match the number of query heads."""
    if repeats == 1:
        return hidden_states
    batch, heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, heads, repeats, seq_len, head_dim)
    return expanded.reshape(batch, heads * repeats, seq_len, head_dim)


class GroupedQueryAttention(nn.Module):
    """Apply grouped-query self-attention with cache support."""

    def __init__(self, config: ModelConfig):
        """Create one grouped-query attention block."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_groups = self.num_heads // self.num_key_value_heads
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = config.attention_dropout
        self._flash_fn = self._load_flash_attention()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run causal attention and return output plus updated cache."""
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        q, k_expanded = apply_rope(q, repeat_kv(k, self.num_groups), cos, sin)
        k = k_expanded[:, :: self.num_groups, :, :]
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)
        expanded_k = repeat_kv(k, self.num_groups)
        expanded_v = repeat_kv(v, self.num_groups)
        if self._flash_fn is not None and hidden_states.is_cuda:
            attn_output = self._flash_attention(q, expanded_k, expanded_v)
        else:
            attn_output = F.scaled_dot_product_attention(
                q,
                expanded_k,
                expanded_v,
                is_causal=past_key_value is None,
                dropout_p=self.dropout if self.training else 0.0,
            )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output), (k, v)

    def _flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Run FlashAttention when it is available."""
        q_flash = q.transpose(1, 2)
        k_flash = k.transpose(1, 2)
        v_flash = v.transpose(1, 2)
        return self._flash_fn(q_flash, k_flash, v_flash, causal=True).transpose(1, 2)

    def _load_flash_attention(self):
        """Load the FlashAttention function when available."""
        try:
            from flash_attn import flash_attn_func
        except ImportError:
            return None
        return flash_attn_func
