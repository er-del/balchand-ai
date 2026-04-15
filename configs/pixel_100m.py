"""100M-class PIXEL preset for CPU smoke testing and local demos."""

from __future__ import annotations

from configs.base import ModelConfig, TrainingConfig


def build_model_config() -> ModelConfig:
    """Return the 100M-class model configuration."""
    return ModelConfig(
        name="pixel_100m",
        vocab_size=16_000,
        context_length=1_024,
        num_layers=12,
        hidden_size=768,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=2_048,
    )


def build_training_config() -> TrainingConfig:
    """Return the default training settings for the 100M preset."""
    return TrainingConfig(
        size="100m",
        output_dir="checkpoints/pixel_100m",
        total_steps=10,
        sequence_length=32,
        batch_size=1,
        grad_accumulation_steps=2,
    )
