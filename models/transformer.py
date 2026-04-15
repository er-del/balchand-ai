"""Decoder-only transformer language model for PIXEL."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn

from configs.base import ModelConfig
from models.block import TransformerBlock
from models.lora import inject_lora
from models.norms import RMSNorm
from models.rope import build_rope_cache


@dataclass(slots=True)
class PixelModelOutput:
    """Return logits, cache, and optional auxiliary losses."""

    logits: torch.Tensor
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]]
    aux_loss: torch.Tensor


class PixelForCausalLM(nn.Module):
    """A GPT-style decoder-only transformer used by PIXEL."""

    def __init__(self, config: ModelConfig):
        """Create the causal language model."""
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(TransformerBlock(config, index) for index in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self._reset_parameters()
        inject_lora(self, config.lora)

    def _reset_parameters(self) -> None:
        """Apply stable default initialization."""
        embed_std = 1.0 / math.sqrt(self.config.hidden_size)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=embed_std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> PixelModelOutput:
        """Run a forward pass and return logits plus updated cache."""
        hidden_states = self.embed_tokens(input_ids)
        _, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        past_key_values = past_key_values or [None] * len(self.layers)
        start = past_key_values[0][0].size(-2) if past_key_values and past_key_values[0] is not None else 0
        cos, sin = build_rope_cache(start + seq_len, self.config.head_dim, self.config.rope_base, device=device)
        cos = cos[start : start + seq_len]
        sin = sin[start : start + seq_len]
        next_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        aux_loss = hidden_states.new_zeros(())
        for layer, past in zip(self.layers, past_key_values):
            hidden_states, present, layer_aux = layer(hidden_states, cos, sin, past)
            next_cache.append(present)
            aux_loss = aux_loss + layer_aux
        logits = self.lm_head(self.norm(hidden_states))
        return PixelModelOutput(logits=logits, past_key_values=next_cache, aux_loss=aux_loss)
