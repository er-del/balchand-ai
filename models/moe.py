"""Optional top-k mixture-of-experts blocks for PIXEL."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from configs.base import ModelConfig


class ExpertMLP(nn.Module):
    """One expert feed-forward network."""

    def __init__(self, config: ModelConfig):
        """Create one expert."""
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the expert feed-forward network."""
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MoELayer(nn.Module):
    """A simple top-k routed MoE layer."""

    def __init__(self, config: ModelConfig):
        """Create the router and experts for one layer."""
        super().__init__()
        self.num_experts = config.moe.num_experts
        self.top_k = config.moe.top_k
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(ExpertMLP(config) for _ in range(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route each token through the top-k experts and return the auxiliary loss."""
        router_logits = self.router(hidden_states)
        top_values, top_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        top_weights = torch.softmax(top_values, dim=-1)
        output = torch.zeros_like(hidden_states)
        for expert_index, expert in enumerate(self.experts):
            mask = top_indices == expert_index
            if not mask.any():
                continue
            weight = torch.where(mask, top_weights, torch.zeros_like(top_weights)).sum(dim=-1, keepdim=True)
            expert_out = expert(hidden_states)
            output = output + expert_out * weight
        router_probs = torch.softmax(router_logits, dim=-1).mean(dim=(0, 1))
        usage = torch.zeros(self.num_experts, device=hidden_states.device, dtype=hidden_states.dtype)
        usage.scatter_add_(0, top_indices.reshape(-1), torch.ones_like(top_indices.reshape(-1), dtype=hidden_states.dtype))
        usage = usage / usage.sum().clamp_min(1.0)
        aux_loss = (router_probs * usage).sum() * float(self.num_experts)
        return output, aux_loss
