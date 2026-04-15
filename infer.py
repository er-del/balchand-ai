"""Flat inference entrypoint for PIXEL."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from configs.registry import get_preset, list_presets
from core.checkpoint import CheckpointInspection, CheckpointManager
from core.types import GenerationRequest
from inference.generator import PixelGenerator
from tokenizer.manager import ensure_tokenizer
from training.bootstrap import ensure_bootstrap_corpus


def build_argparser() -> argparse.ArgumentParser:
    """Build the PIXEL inference CLI."""
    parser = argparse.ArgumentParser(description="Run inference with a PIXEL model.")
    parser.add_argument("--prompt", default="Write a short paragraph about reliable local AI tooling.", help="Prompt text to generate from.")
    parser.add_argument("--model", default=None, help="Checkpoint file or directory to load.")
    parser.add_argument(
        "--size",
        default="100m",
        choices=list_presets(),
        help="Model size preset used only when no compatible checkpoint is loaded.",
    )
    parser.add_argument("--max-tokens", type=int, default=96, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature. Use 0 for greedy decoding.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling cutoff.")
    parser.add_argument("--mode", default="chat", choices=["chat", "completion", "summarize", "code"], help="Inference UI mode.")
    return parser


def _latest_checkpoint() -> str | None:
    """Find the newest PIXEL checkpoint file, if any exist."""
    candidates = sorted(Path("checkpoints").glob("*/latest.pt"))
    if not candidates:
        return None
    return str(candidates[-1])


def _inspect_checkpoint(checkpoint: str | None) -> CheckpointInspection | None:
    """Inspect a checkpoint file or directory when one is available."""
    if checkpoint is None:
        return None
    target = Path(checkpoint)
    manager = CheckpointManager(target.parent if target.suffix == ".pt" else target, create=False)
    return manager.inspect(path=checkpoint, device="cpu")


def main() -> None:
    """Run PIXEL inference from the command line."""
    args = build_argparser().parse_args()
    model_config, _ = get_preset(args.size)
    checkpoint = args.model or _latest_checkpoint()
    checkpoint_info = _inspect_checkpoint(checkpoint)
    data_path = str(ensure_bootstrap_corpus())
    tokenizer_vocab = checkpoint_info.model_config.vocab_size if checkpoint_info is not None else min(model_config.vocab_size, 4096)
    tokenizer = ensure_tokenizer(data_paths=[data_path], vocab_size=tokenizer_vocab)
    if checkpoint_info is None:
        model_config.vocab_size = tokenizer.vocab_size
    generator = PixelGenerator(
        model_config,
        tokenizer,
        checkpoint_path=checkpoint,
        checkpoint_info=checkpoint_info,
    )
    request = GenerationRequest(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        mode=args.mode,
    )
    response = generator.generate(request)
    payload = asdict(response)
    payload.update(generator.describe())
    if not response.used_checkpoint:
        payload["warning"] = "No trained checkpoint was loaded. Output comes from randomly initialized weights."
    elif generator.config_source == "checkpoint":
        payload["note"] = (
            f"Checkpoint metadata overrode the requested preset '{args.size}'."
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
