"""Flat training entrypoint for PIXEL."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from configs.registry import get_preset, list_presets
from tokenizer.manager import ensure_tokenizer
from training.bootstrap import ensure_bootstrap_corpus
from training.trainer import train_model


def build_argparser() -> argparse.ArgumentParser:
    """Build the PIXEL training CLI."""
    parser = argparse.ArgumentParser(description="Train a PIXEL language model with sensible defaults.")
    parser.add_argument("--size", default="100m", choices=list_presets(), help="Model size preset to train.")
    parser.add_argument("--data", default=None, help="Training data file path. Supports .txt, .jsonl, or .parquet.")
    parser.add_argument("--output", default=None, help="Checkpoint directory. Defaults to the preset path.")
    parser.add_argument("--steps", type=int, default=None, help="Number of optimizer steps to run.")
    parser.add_argument("--mode", default="pretrain", choices=["pretrain", "lora", "qlora"], help="Training mode to run.")
    parser.add_argument("--checkpoint", default=None, help="Existing checkpoint path reserved for future resume flows.")
    parser.add_argument("--use-moe", action="store_true", help="Enable Mixture of Experts when the preset supports it.")
    return parser


def _maybe_launch_distributed() -> None:
    """Relaunch the trainer under torch distributed when multiple GPUs are visible."""
    if os.environ.get("WORLD_SIZE"):
        return
    try:
        import torch
    except ImportError:
        return
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count <= 1:
        return
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={gpu_count}",
        Path(__file__).name,
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(command))


def main() -> None:
    """Run the PIXEL training workflow."""
    _maybe_launch_distributed()
    args = build_argparser().parse_args()
    model_config, training_config = get_preset(args.size, use_moe=args.use_moe)
    if args.output:
        training_config.output_dir = args.output
    if args.steps is not None:
        training_config.total_steps = args.steps
    training_config.mode = args.mode
    if args.mode in {"lora", "qlora"}:
        model_config.lora.enabled = True
    data_path = args.data or str(ensure_bootstrap_corpus())
    training_config.data_path = data_path
    tokenizer = ensure_tokenizer(data_paths=[data_path], vocab_size=min(model_config.vocab_size, 4096))
    model_config.vocab_size = tokenizer.vocab_size
    summary = train_model(model_config, training_config, tokenizer)
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
