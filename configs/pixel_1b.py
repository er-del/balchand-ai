"""1B-class PIXEL preset for light GPU training."""

from __future__ import annotations

from configs.base import ModelConfig, TrainingConfig


def build_model_config() -> ModelConfig:
    """Return the 1B-class model configuration."""
    return ModelConfig(
        name="pixel_1b",
        vocab_size=32_000,
        context_length=4_096,
        num_layers=24,
        hidden_size=2_048,
        num_attention_heads=16,
        num_key_value_heads=8,
        intermediate_size=5_632,
    )


def build_training_config() -> TrainingConfig:
    """Return the default training settings for the 1B preset."""
    return TrainingConfig(
        size="1b",
        output_dir="checkpoints/pixel_1b",
        total_steps=200,
        sequence_length=512,
        batch_size=1,
        grad_accumulation_steps=8,
        gradient_checkpointing=True,
    )
