# configs

This folder contains typed preset definitions used by `train.py` and `infer.py`.

## Why It Is Structured This Way

- No YAML parsing is required for normal use.
- All presets are Python dataclasses with explicit fields.
- CLI size flags map directly to registry entries.

## Preset Names

- `100m`
- `1b`
- `3b`
- `7b`

## Related Commands

```bash
python train.py --size 100m
python train.py --size 1b
python train.py --size 3b --use-moe
python infer.py --size 7b --prompt "Summarize this..."
```

## Key Files

- `base.py`: shared config dataclasses (`ModelConfig`, `TrainingConfig`, `RuntimeConfig`)
- `registry.py`: size-name to preset resolver used by CLI and web code
- `pixel_100m.py`, `pixel_1b.py`, `pixel_3b.py`, `pixel_7b.py`: concrete preset builders
