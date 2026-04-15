"""7B-class PIXEL preset for high-end GPU or multi-GPU training."""

from __future__ import annotations

from configs.base import ModelConfig, MoEConfig, TrainingConfig


def build_model_config(use_moe: bool = False) -> ModelConfig:
    """Return the 7B-class model configuration."""
    return ModelConfig(
        name="pixel_7b",
        vocab_size=32_000,
        context_length=8_192,
        num_layers=32,
        hidden_size=4_096,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=11_008,
        use_moe=use_moe,
        moe=MoEConfig(enabled=use_moe, num_experts=8, top_k=2, expert_interval=4),
    )


def build_training_config() -> TrainingConfig:
    """Return the default training settings for the 7B preset."""
    return TrainingConfig(
        size="7b",
        output_dir="checkpoints/pixel_7b",
        total_steps=600,
        sequence_length=2_048,
        batch_size=1,
        grad_accumulation_steps=32,
        gradient_checkpointing=True,
    )
