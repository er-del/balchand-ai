"""Install PIXEL dependencies and print a hardware-aware setup summary."""

from __future__ import annotations

import subprocess
import sys

from core.runtime import RuntimeManager


def _run_pip(args: list[str]) -> int:
    """Run one pip command and return its exit code."""
    command = [sys.executable, "-m", "pip", *args]
    return subprocess.call(command)


def main() -> None:
    """Install required packages and print the detected runtime profile."""
    runtime = RuntimeManager()
    print("Installing PIXEL core requirements...")
    _run_pip(["install", "-r", "requirements.txt"])
    hardware = runtime.detect_hardware()
    if hardware.device == "cuda" and hardware.supports_flash_attention:
        print("FlashAttention is already available.")
    elif hardware.device == "cuda":
        print("FlashAttention is optional and was not detected.")
    if hardware.device == "cuda" and hardware.supports_bitsandbytes:
        print("bitsandbytes is already available.")
    elif hardware.device == "cuda":
        print("bitsandbytes is optional and was not detected.")
    print("PIXEL hardware summary:")
    for key, value in runtime.health_payload()["hardware"].items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
