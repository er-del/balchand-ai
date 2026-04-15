"""LoRA adapter layers for PIXEL."""

from __future__ import annotations

import math

import torch
from torch import nn

from configs.base import LoRAConfig


class LoRALinear(nn.Module):
    """Wrap a linear layer with a trainable low-rank update."""

    def __init__(self, base_layer: nn.Linear, config: LoRAConfig):
        """Create one LoRA-wrapped linear layer."""
        super().__init__()
        self.base_layer = base_layer
        self.rank = config.rank
        self.scale = config.alpha / max(config.rank, 1)
        self.dropout = nn.Dropout(config.dropout)
        self.lora_a = nn.Parameter(torch.zeros(config.rank, base_layer.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base_layer.out_features, config.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the frozen base layer plus the LoRA update."""
        base = self.base_layer(inputs)
        update = self.dropout(inputs) @ self.lora_a.t()
        update = update @ self.lora_b.t()
        return base + update * self.scale


def inject_lora(module: nn.Module, config: LoRAConfig) -> None:
    """Replace selected linear submodules with LoRA-wrapped versions."""
    if not config.enabled:
        return
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in config.target_modules:
            setattr(module, name, LoRALinear(child, config))
        else:
            inject_lora(child, config)
