# scripts

This folder contains optional helper scripts that are not required for normal setup/train/infer/web usage.

## Legacy Import Script

Import selected assets from an older SAGE checkout:

```bash
python scripts/import_legacy_sage.py --legacy-root "C:\path\to\sage" --copy-tokenizer --copy-data
```

Copy only checkpoints:

```bash
python scripts/import_legacy_sage.py --legacy-root "C:\path\to\sage" --copy-checkpoints
```

## What It Copies

- Tokenizer files into `tokenizer/`
- Raw JSONL data into `data/imported/`
- Checkpoints into `artifacts/imported_checkpoints/`
