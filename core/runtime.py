"""Runtime policy and hardware detection for PIXEL."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
import platform
from pathlib import Path
from typing import Any

import torch

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

from configs.base import RuntimeConfig, TrainingConfig


@dataclass(slots=True)
class HardwareProfile:
    """Describe the visible hardware and runtime decisions."""

    device: str
    device_index: int
    dtype: torch.dtype
    gpu_count: int
    total_ram_gb: float
    total_vram_gb: float
    supports_flash_attention: bool
    supports_bitsandbytes: bool
    distributed: bool
    gradient_checkpointing: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize the hardware profile for logs and APIs."""
        return {
            "device": self.device,
            "device_index": self.device_index,
            "dtype": str(self.dtype),
            "gpu_count": self.gpu_count,
            "total_ram_gb": round(self.total_ram_gb, 2),
            "total_vram_gb": round(self.total_vram_gb, 2),
            "supports_flash_attention": self.supports_flash_attention,
            "supports_bitsandbytes": self.supports_bitsandbytes,
            "distributed": self.distributed,
            "gradient_checkpointing": self.gradient_checkpointing,
        }


class RuntimeManager:
    """Resolve paths, hardware, and runtime defaults for PIXEL."""

    def __init__(self, project_root: str | Path | None = None):
        """Create a runtime manager for one PIXEL checkout."""
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()

    def resolve(self, *parts: str) -> Path:
        """Resolve a path relative to the project root."""
        return self.project_root.joinpath(*parts)

    def detect_hardware(self, runtime: RuntimeConfig | None = None, training: TrainingConfig | None = None) -> HardwareProfile:
        """Detect the best available device and precision policy."""
        runtime = runtime or RuntimeConfig()
        training = training or TrainingConfig()
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        distributed = gpu_count > 1
        device = "cpu"
        device_index = -1
        dtype = torch.float32
        total_vram_gb = 0.0
        if runtime.device == "cpu":
            pass
        elif torch.cuda.is_available() and runtime.device in {"auto", "cuda"}:
            device = "cuda"
            device_index = int(os.environ.get("LOCAL_RANK", "0")) if distributed else 0
            props = torch.cuda.get_device_properties(device_index)
            total_vram_gb = props.total_memory / (1024 ** 3)
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        elif torch.backends.mps.is_available() and runtime.device in {"auto", "mps"}:
            device = "mps"
            dtype = torch.float16
        total_ram_gb = self._detect_system_ram_gb()
        supports_flash_attention = bool(device == "cuda" and self._module_available("flash_attn"))
        supports_bitsandbytes = bool(device == "cuda" and self._module_available("bitsandbytes"))
        gradient_checkpointing = bool(training.gradient_checkpointing or (device == "cuda" and total_vram_gb < runtime.gradient_checkpoint_vram_gb))
        return HardwareProfile(
            device=device,
            device_index=device_index,
            dtype=dtype,
            gpu_count=gpu_count,
            total_ram_gb=total_ram_gb,
            total_vram_gb=total_vram_gb,
            supports_flash_attention=supports_flash_attention,
            supports_bitsandbytes=supports_bitsandbytes,
            distributed=distributed,
            gradient_checkpointing=gradient_checkpointing,
        )

    def build_device(self, hardware: HardwareProfile) -> torch.device:
        """Create a torch device from a hardware profile."""
        if hardware.device == "cuda":
            return torch.device(f"cuda:{hardware.device_index}")
        return torch.device(hardware.device)

    def available_checkpoints(self) -> list[str]:
        """List available checkpoint directories."""
        root = self.resolve("checkpoints")
        if not root.exists():
            return []
        return sorted(path.name for path in root.iterdir() if path.is_dir())

    def health_payload(self) -> dict[str, Any]:
        """Build a JSON-safe health payload for CLI and web use."""
        hardware = self.detect_hardware()
        return {
            "status": "ok",
            "hardware": hardware.to_dict(),
            "checkpoints": self.available_checkpoints(),
        }

    def _detect_system_ram_gb(self) -> float:
        """Detect installed system RAM in gigabytes."""
        if psutil is not None:
            return psutil.virtual_memory().total / (1024 ** 3)
        if platform.system() == "Windows":
            kernel32 = ctypes.windll.kernel32
            mem_kb = ctypes.c_ulonglong()
            kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(mem_kb))
            return (mem_kb.value * 1024) / (1024 ** 3)
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / (1024 ** 3)
        return 0.0

    def _module_available(self, name: str) -> bool:
        """Return True when an optional module can be imported."""
        try:
            __import__(name)
        except ImportError:
            return False
        return True
