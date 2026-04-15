"""End-to-end smoke test for PIXEL."""

from __future__ import annotations

import sys

import torch

from configs.registry import get_preset
from core.runtime import RuntimeManager
from core.types import GenerationRequest, SmokeTestResult
from inference.generator import PixelGenerator
from models.transformer import PixelForCausalLM
from tokenizer.manager import ensure_tokenizer
from training.bootstrap import ensure_bootstrap_corpus
from training.trainer import train_model


def run_smoke_test() -> SmokeTestResult:
    """Run the contracted PIXEL smoke test workflow."""
    details: list[str] = []
    runtime = RuntimeManager()
    hardware = runtime.detect_hardware()
    details.append(f"hardware: {hardware.to_dict()}")
    data_path = str(ensure_bootstrap_corpus())
    model_config, training_config = get_preset("100m")
    training_config.data_path = data_path
    training_config.total_steps = 3
    training_config.save_every = 3
    training_config.sequence_length = 16
    training_config.grad_accumulation_steps = 1
    training_config.output_dir = "checkpoints/pixel_smoke"
    tokenizer = ensure_tokenizer(data_paths=[data_path], vocab_size=2048)
    model_config.vocab_size = tokenizer.vocab_size
    model = PixelForCausalLM(model_config)
    input_ids = torch.randint(0, model_config.vocab_size, (1, 8))
    output = model(input_ids)
    assert output.logits.shape == (1, 8, model_config.vocab_size)
    details.append("forward pass: ok")
    summary = train_model(model_config, training_config, tokenizer)
    details.append(f"training: {summary.steps_completed} steps")
    generator = PixelGenerator(model_config, tokenizer, checkpoint_path="checkpoints/pixel_smoke/latest.pt")
    response = generator.generate(GenerationRequest(prompt="PIXEL", max_tokens=8, temperature=0.0, top_p=1.0))
    assert isinstance(response.output, str)
    assert response.tokens_generated >= 0
    details.append("inference: ok")
    return SmokeTestResult(success=True, details=details)


def main() -> None:
    """Run the PIXEL smoke test and exit with a shell-friendly status code."""
    try:
        result = run_smoke_test()
    except Exception as exc:  # pragma: no cover - CLI failure path
        print("PIXEL smoke test failed")
        print(str(exc))
        raise SystemExit(1) from exc
    print("PIXEL smoke test passed")
    for line in result.details:
        print(f"- {line}")
    raise SystemExit(0 if result.success else 1)


if __name__ == "__main__":
    main()
