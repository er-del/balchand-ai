"""Tests for PIXEL model behavior."""

import torch

from configs.registry import get_preset
from models.transformer import PixelForCausalLM


def test_forward_shape_and_weight_tying() -> None:
    model_config, _ = get_preset("100m")
    model_config.vocab_size = 256
    model = PixelForCausalLM(model_config)
    input_ids = torch.randint(0, model_config.vocab_size, (2, 8))
    output = model(input_ids)
    assert output.logits.shape == (2, 8, model_config.vocab_size)
    assert len(output.past_key_values) == model_config.num_layers
    assert model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
