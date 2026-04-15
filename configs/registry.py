"""Preset registry for PIXEL model sizes."""

from __future__ import annotations

from typing import Callable

from configs.base import ModelConfig, TrainingConfig
from configs import pixel_100m, pixel_1b, pixel_3b, pixel_7b


PresetBuilder = tuple[Callable[..., ModelConfig], Callable[[], TrainingConfig]]


PRESETS: dict[str, PresetBuilder] = {
    "100m": (pixel_100m.build_model_config, pixel_100m.build_training_config),
    "1b": (pixel_1b.build_model_config, pixel_1b.build_training_config),
    "3b": (pixel_3b.build_model_config, pixel_3b.build_training_config),
    "7b": (pixel_7b.build_model_config, pixel_7b.build_training_config),
}


def list_presets() -> list[str]:
    """Return the supported preset identifiers."""
    return sorted(PRESETS)


def get_preset(name: str, use_moe: bool = False) -> tuple[ModelConfig, TrainingConfig]:
    """Resolve one preset into a model and training config pair."""
    key = name.lower()
    if key not in PRESETS:
        raise KeyError(f"Unknown PIXEL preset: {name}")
    model_builder, train_builder = PRESETS[key]
    model = model_builder(use_moe=use_moe) if key in {"3b", "7b"} else model_builder()
    train = train_builder()
    return model, train
