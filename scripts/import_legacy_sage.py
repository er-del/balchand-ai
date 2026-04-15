"""Import useful legacy SAGE assets into PIXEL without reusing its architecture."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    """Build the legacy import CLI."""
    parser = argparse.ArgumentParser(description="Import useful assets from a legacy SAGE checkout.")
    parser.add_argument("--legacy-root", required=True, help="Path to the legacy SAGE repository root.")
    parser.add_argument("--copy-tokenizer", action="store_true", help="Copy tokenizer model and vocab into PIXEL.")
    parser.add_argument("--copy-data", action="store_true", help="Copy raw JSONL corpus files into PIXEL data/imported.")
    parser.add_argument("--copy-checkpoints", action="store_true", help="Copy checkpoint files into PIXEL artifacts/imported_checkpoints.")
    return parser


def _copy_if_exists(source: Path, target: Path) -> bool:
    """Copy one file when it exists."""
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def main() -> None:
    """Run the legacy import workflow."""
    args = build_argparser().parse_args()
    legacy_root = Path(args.legacy_root).resolve()
    summary = {"legacy_root": str(legacy_root), "copied": []}
    if args.copy_tokenizer:
        for name in ("tokenizer.model", "tokenizer.vocab"):
            source = legacy_root / "tokenizer" / name
            target = Path("tokenizer") / f"legacy_{name}"
            if _copy_if_exists(source, target):
                summary["copied"].append(str(target))
    if args.copy_data:
        raw_dir = legacy_root / "data" / "raw"
        if raw_dir.exists():
            for item in raw_dir.glob("*.jsonl"):
                target = Path("data") / "imported" / item.name
                if _copy_if_exists(item, target):
                    summary["copied"].append(str(target))
    if args.copy_checkpoints:
        runs_dir = legacy_root / "runs"
        if runs_dir.exists():
            for item in runs_dir.glob("**/*.pt"):
                target = Path("artifacts") / "imported_checkpoints" / item.name
                if _copy_if_exists(item, target):
                    summary["copied"].append(str(target))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
