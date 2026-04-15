"""3B-class PIXEL preset for mid-range GPU or multi-GPU training."""

from __future__ import annotations

from configs.base import ModelConfig, MoEConfig, TrainingConfig


def build_model_config(use_moe: bool = False) -> ModelConfig:
    """Return the 3B-class model configuration."""
    return ModelConfig(
        name="pixel_3b",
        vocab_size=32_000,
        context_length=4_096,
        num_layers=32,
        hidden_size=3_072,
        num_attention_heads=24,
        num_key_value_heads=8,
        intermediate_size=8_192,
        use_moe=use_moe,
        moe=MoEConfig(enabled=use_moe, num_experts=4, top_k=2, expert_interval=4),
    )


def build_training_config() -> TrainingConfig:
    """Return the default training settings for the 3B preset."""
    return TrainingConfig(
        size="3b",
        output_dir="checkpoints/pixel_3b",
        total_steps=400,
        sequence_length=1_024,
        batch_size=1,
        grad_accumulation_steps=16,
        gradient_checkpointing=True,
    )
