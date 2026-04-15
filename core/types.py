"""Shared request and response types for PIXEL scripts and services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GenerationRequest:
    """Describe one text generation request.

    Args:
        prompt: Input prompt text.
        max_tokens: Maximum number of newly generated tokens.
        temperature: Sampling temperature.
        top_p: Nucleus sampling cutoff.
        mode: UX mode used by the CLI or web UI.
    """

    prompt: str
    max_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    mode: str = "chat"


@dataclass(slots=True)
class GenerationResponse:
    """Return generation results in a uniform shape.

    Args:
        output: Generated text.
        tokens_generated: Count of generated tokens.
        model_name: Model preset or checkpoint label used for generation.
        used_checkpoint: Whether generation used learned weights.
    """

    output: str
    tokens_generated: int
    model_name: str
    used_checkpoint: bool


@dataclass(slots=True)
class TrainSummary:
    """Summarize one training run.

    Args:
        output_dir: Directory where checkpoints were written.
        steps_completed: Number of optimizer steps completed.
        loss_history: Sample of observed losses.
        hardware: JSON-safe hardware summary.
    """

    output_dir: str
    steps_completed: int
    loss_history: list[float] = field(default_factory=list)
    hardware: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SmokeTestResult:
    """Represent the smoke test outcome.

    Args:
        success: True when every smoke test step passed.
        details: Human-readable status lines.
    """

    success: bool
    details: list[str]


@dataclass(slots=True)
class HealthResponse:
    """Return runtime health details for the web app.

    Args:
        status: Health string, usually `ok`.
        hardware: Hardware profile summary.
        checkpoints: Available checkpoint names.
    """

    status: str
    hardware: dict[str, Any]
    checkpoints: list[str]
