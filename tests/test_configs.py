"""Tests for PIXEL config presets."""

from configs.registry import get_preset, list_presets


def test_list_presets_contains_expected_sizes() -> None:
    assert list_presets() == ["100m", "1b", "3b", "7b"]


def test_100m_preset_has_reasonable_shape() -> None:
    model, train = get_preset("100m")
    assert model.name == "pixel_100m"
    assert model.hidden_size == 768
    assert train.output_dir.endswith("pixel_100m")
