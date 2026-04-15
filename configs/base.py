"""Dataclass-based configuration objects for PIXEL."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class LoRAConfig:
    """Define LoRA adapter settings."""

    enabled: bool = False
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "LoRAConfig":
        """Build a LoRA config from a serialized checkpoint payload."""
        if not payload:
            return cls()
        target_modules = payload.get("target_modules", cls().target_modules)
        if isinstance(target_modules, list):
            target_modules = tuple(str(item) for item in target_modules)
        return cls(
            enabled=bool(payload.get("enabled", False)),
            rank=int(payload.get("rank", 8)),
            alpha=int(payload.get("alpha", 16)),
            dropout=float(payload.get("dropout", 0.05)),
            target_modules=tuple(target_modules),
        )


@dataclass(slots=True)
class MoEConfig:
    """Define optional mixture-of-experts settings."""

    enabled: bool = False
    num_experts: int = 4
    top_k: int = 2
    expert_interval: int = 4

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "MoEConfig":
        """Build an MoE config from a serialized checkpoint payload."""
        if not payload:
            return cls()
        return cls(
            enabled=bool(payload.get("enabled", False)),
            num_experts=int(payload.get("num_experts", 4)),
            top_k=int(payload.get("top_k", 2)),
            expert_interval=int(payload.get("expert_interval", 4)),
        )


@dataclass(slots=True)
class ModelConfig:
    """Describe one PIXEL transformer preset."""

    name: str
    vocab_size: int
    context_length: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    rope_base: int = 500_000
    tie_word_embeddings: bool = True
    dropout: float = 0.0
    attention_dropout: float = 0.0
    rms_norm_eps: float = 1.0e-5
    use_moe: bool = False
    moe: MoEConfig = field(default_factory=MoEConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    @property
    def head_dim(self) -> int:
        """Return the per-head hidden size."""
        return self.hidden_size // self.num_attention_heads

    @property
    def approx_parameters(self) -> int:
        """Return an approximate parameter count for reporting."""
        embeddings = self.vocab_size * self.hidden_size
        attn = self.num_layers * self.hidden_size * (self.hidden_size * 2 + self.num_key_value_heads * self.head_dim * 2)
        mlp = self.num_layers * self.hidden_size * self.intermediate_size * 3
        return embeddings + attn + mlp

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ModelConfig":
        """Build a model config from serialized checkpoint metadata."""
        return cls(
            name=str(payload["name"]),
            vocab_size=int(payload["vocab_size"]),
            context_length=int(payload["context_length"]),
            num_layers=int(payload["num_layers"]),
            hidden_size=int(payload["hidden_size"]),
            num_attention_heads=int(payload["num_attention_heads"]),
            num_key_value_heads=int(payload["num_key_value_heads"]),
            intermediate_size=int(payload["intermediate_size"]),
            rope_base=int(payload.get("rope_base", 500_000)),
            tie_word_embeddings=bool(payload.get("tie_word_embeddings", True)),
            dropout=float(payload.get("dropout", 0.0)),
            attention_dropout=float(payload.get("attention_dropout", 0.0)),
            rms_norm_eps=float(payload.get("rms_norm_eps", 1.0e-5)),
            use_moe=bool(payload.get("use_moe", False)),
            moe=MoEConfig.from_dict(payload.get("moe") if isinstance(payload.get("moe"), dict) else None),
            lora=LoRAConfig.from_dict(payload.get("lora") if isinstance(payload.get("lora"), dict) else None),
        )


@dataclass(slots=True)
class TrainingConfig:
    """Describe training defaults for a PIXEL run."""

    size: str = "100m"
    data_path: str = "data/bootstrap/demo_corpus.txt"
    output_dir: str = "checkpoints/pixel_100m"
    batch_size: int = 1
    grad_accumulation_steps: int = 4
    learning_rate: float = 3.0e-4
    min_learning_rate: float = 3.0e-5
    warmup_steps: int = 4
    total_steps: int = 25
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    eval_every: int = 10
    save_every: int = 10
    seed: int = 42
    gradient_checkpointing: bool = False
    mode: str = "pretrain"
    sequence_length: int = 64
    num_workers: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "TrainingConfig":
        """Build a training config from serialized checkpoint metadata."""
        if not payload:
            return cls()
        return cls(
            size=str(payload.get("size", "100m")),
            data_path=str(payload.get("data_path", "data/bootstrap/demo_corpus.txt")),
            output_dir=str(payload.get("output_dir", "checkpoints/pixel_100m")),
            batch_size=int(payload.get("batch_size", 1)),
            grad_accumulation_steps=int(payload.get("grad_accumulation_steps", 4)),
            learning_rate=float(payload.get("learning_rate", 3.0e-4)),
            min_learning_rate=float(payload.get("min_learning_rate", 3.0e-5)),
            warmup_steps=int(payload.get("warmup_steps", 4)),
            total_steps=int(payload.get("total_steps", 25)),
            weight_decay=float(payload.get("weight_decay", 0.1)),
            max_grad_norm=float(payload.get("max_grad_norm", 1.0)),
            eval_every=int(payload.get("eval_every", 10)),
            save_every=int(payload.get("save_every", 10)),
            seed=int(payload.get("seed", 42)),
            gradient_checkpointing=bool(payload.get("gradient_checkpointing", False)),
            mode=str(payload.get("mode", "pretrain")),
            sequence_length=int(payload.get("sequence_length", 64)),
            num_workers=int(payload.get("num_workers", 0)),
        )


@dataclass(slots=True)
class RuntimeConfig:
    """Describe runtime and hardware policy for PIXEL."""

    device: str = "auto"
    gradient_checkpoint_vram_gb: float = 20.0
    enable_flash_attention: bool = True
    enable_bitsandbytes: bool = True
