"""
pixel/hf_push.py

Prepare and optionally upload a PIXEL model bundle to the Hugging Face Hub.

This script is intentionally strict about what belongs on Hugging Face:
- model checkpoint files
- tokenizer assets
- generated model card
- compact exported metadata

It does not upload the full PIXEL source tree. Source code and docs belong in GitHub.

Usage:
    python hf_push.py
    python hf_push.py --repo-id username/pixel-100m
    python hf_push.py --checkpoint checkpoints/pixel_100m/latest.pt --repo-id username/pixel-100m
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from core.checkpoint import CheckpointInspection, CheckpointManager

try:
    from huggingface_hub import HfApi
except ImportError:  # pragma: no cover - optional dependency for upload only
    HfApi = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "pixel_100m" / "latest.pt"
DEFAULT_TOKENIZER_PREFIX = PROJECT_ROOT / "tokenizer" / "pixel_tokenizer"
DEFAULT_EXPORT_ROOT = PROJECT_ROOT / "artifacts" / "hf_export"


def build_argparser() -> argparse.ArgumentParser:
    """Build the Hugging Face packaging CLI."""
    parser = argparse.ArgumentParser(
        description="Prepare a PIXEL model bundle for Hugging Face and optionally upload it."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Checkpoint file or checkpoint directory to package.",
    )
    parser.add_argument(
        "--tokenizer-prefix",
        default=str(DEFAULT_TOKENIZER_PREFIX),
        help="Tokenizer prefix without extension, for example tokenizer/pixel_tokenizer.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face repo id to upload to, for example username/pixel-100m.",
    )
    parser.add_argument(
        "--export-dir",
        default=str(DEFAULT_EXPORT_ROOT),
        help="Directory where the Hugging Face-ready bundle will be created.",
    )
    parser.add_argument(
        "--message",
        default="Upload PIXEL model bundle",
        help="Commit message used when uploading to Hugging Face.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hugging Face repo as private when it does not already exist.",
    )
    return parser


def inspect_checkpoint(checkpoint: str | Path) -> CheckpointInspection:
    """Inspect a PIXEL checkpoint and return typed metadata."""
    target = Path(checkpoint)
    manager = CheckpointManager(target.parent if target.suffix == ".pt" else target, create=False)
    inspection = manager.inspect(path=checkpoint, device="cpu")
    if inspection is None:
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint}")
    return inspection


def validate_tokenizer_files(prefix: str | Path) -> tuple[Path, Path]:
    """Validate that tokenizer model and vocab files exist."""
    base = Path(prefix)
    model_path = base.with_suffix(".model")
    vocab_path = base.with_suffix(".vocab")
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model file was not found: {model_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Tokenizer vocab file was not found: {vocab_path}")
    return model_path, vocab_path


def sanitize_repo_name(repo_id: str | None, inspection: CheckpointInspection) -> str:
    """Return a folder-safe export directory name."""
    if repo_id:
        return repo_id.replace("/", "__")
    return inspection.model_config.name


def build_model_card(
    inspection: CheckpointInspection,
    tokenizer_model: Path,
    tokenizer_vocab: Path,
    checkpoint_path: Path,
) -> str:
    """Generate a Hugging Face model card for one PIXEL checkpoint."""
    model = inspection.model_config
    hardware = inspection.metadata.get("hardware", {})
    training = inspection.training_config
    training_lines = [
        f"- Training size preset: `{training.size}`" if training is not None else "- Training size preset: unknown",
        f"- Total steps saved in checkpoint: `{inspection.step}`",
        f"- Sequence length: `{training.sequence_length}`" if training is not None else "- Sequence length: unknown",
        f"- Batch size: `{training.batch_size}`" if training is not None else "- Batch size: unknown",
        f"- Gradient accumulation: `{training.grad_accumulation_steps}`" if training is not None else "- Gradient accumulation: unknown",
    ]
    hardware_lines = [
        f"- Device: `{hardware.get('device', 'unknown')}`",
        f"- GPU count: `{hardware.get('gpu_count', 'unknown')}`",
        f"- Dtype: `{hardware.get('dtype', 'unknown')}`",
    ]
    return f"""---
language:
- en
license: mit
library_name: pixel
pipeline_tag: text-generation
tags:
- pixel
- causal-lm
- local-llm
- pytorch
---

# {model.name}

This repository contains a PIXEL checkpoint exported for use with the PIXEL codebase. It includes the model checkpoint, tokenizer files, exported config metadata, and this model card.

## What This Model Is

`{model.name}` is a decoder-only transformer checkpoint from the PIXEL project. This bundle is intended to be used with the PIXEL runtime rather than the Transformers `AutoModel` API.

## Architecture

- Approximate parameter class: `~{model.approx_parameters:,}`
- Vocab size: `{model.vocab_size}`
- Context length: `{model.context_length}`
- Layers: `{model.num_layers}`
- Hidden size: `{model.hidden_size}`
- Attention heads: `{model.num_attention_heads}`
- Key/value heads: `{model.num_key_value_heads}`
- Intermediate size: `{model.intermediate_size}`
- RoPE base: `{model.rope_base}`
- Uses MoE: `{model.use_moe}`

## Included Files

- `latest.pt`: PIXEL checkpoint
- `manifest.json`: exported checkpoint pointer
- `pixel_tokenizer.model`: SentencePiece tokenizer model
- `pixel_tokenizer.vocab`: SentencePiece tokenizer vocab
- `pixel_model_config.json`: exported typed model config
- `pixel_training_config.json`: exported training config when available

## Training Snapshot

{chr(10).join(training_lines)}

## Runtime Snapshot

{chr(10).join(hardware_lines)}

## Usage With PIXEL

Clone the PIXEL codebase, place or download this bundle, then run:

```bash
python infer.py --model checkpoints/{model.name}/latest.pt --prompt "Hello from PIXEL"
```

Make sure the checkpoint and tokenizer come from the same export bundle.

## Limitations

- This checkpoint is not guaranteed to be instruction-tuned.
- Output quality depends on the training corpus and training duration used for this run.
- This bundle is PIXEL-specific and is not advertised as a drop-in Transformers checkpoint.

## Export Provenance

- Source checkpoint: `{checkpoint_path.name}`
- Source tokenizer model: `{tokenizer_model.name}`
- Source tokenizer vocab: `{tokenizer_vocab.name}`
"""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON payload with stable formatting."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_export_bundle(
    inspection: CheckpointInspection,
    checkpoint_path: Path,
    tokenizer_model: Path,
    tokenizer_vocab: Path,
    export_root: str | Path,
    repo_id: str | None,
) -> Path:
    """Create a local Hugging Face-ready export directory."""
    export_root_path = Path(export_root)
    target_dir = export_root_path / sanitize_repo_name(repo_id, inspection)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(checkpoint_path, target_dir / "latest.pt")
    manifest_payload = {"latest": "latest.pt", "step": inspection.step}
    write_json(target_dir / "manifest.json", manifest_payload)
    shutil.copy2(tokenizer_model, target_dir / "pixel_tokenizer.model")
    shutil.copy2(tokenizer_vocab, target_dir / "pixel_tokenizer.vocab")
    write_json(target_dir / "pixel_model_config.json", asdict(inspection.model_config))
    if inspection.training_config is not None:
        write_json(target_dir / "pixel_training_config.json", asdict(inspection.training_config))
    model_card = build_model_card(inspection, tokenizer_model, tokenizer_vocab, checkpoint_path)
    (target_dir / "README.md").write_text(model_card, encoding="utf-8")
    return target_dir


def upload_bundle(bundle_dir: Path, repo_id: str, commit_message: str, private: bool) -> None:
    """Upload a prepared bundle directory to the Hugging Face Hub."""
    if HfApi is None:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it with `pip install huggingface_hub` before uploading."
        )
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(bundle_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )


def main() -> None:
    """Prepare a PIXEL Hugging Face bundle and optionally upload it."""
    args = build_argparser().parse_args()
    inspection = inspect_checkpoint(args.checkpoint)
    checkpoint_path = Path(inspection.path)
    tokenizer_model, tokenizer_vocab = validate_tokenizer_files(args.tokenizer_prefix)
    bundle_dir = prepare_export_bundle(
        inspection=inspection,
        checkpoint_path=checkpoint_path,
        tokenizer_model=tokenizer_model,
        tokenizer_vocab=tokenizer_vocab,
        export_root=args.export_dir,
        repo_id=args.repo_id,
    )
    print(f"Hugging Face bundle prepared at: {bundle_dir}")
    if args.repo_id is None:
        print("No --repo-id was provided, so nothing was uploaded.")
        return
    upload_bundle(bundle_dir=bundle_dir, repo_id=args.repo_id, commit_message=args.message, private=args.private)
    print(f"Uploaded PIXEL model bundle to Hugging Face repo: {args.repo_id}")


if __name__ == "__main__":
    main()
