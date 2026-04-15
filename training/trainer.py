"""Training loop implementation for PIXEL."""

from __future__ import annotations

from dataclasses import asdict
import math
import os
import random
from pathlib import Path

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from configs.base import ModelConfig, RuntimeConfig, TrainingConfig
from core.checkpoint import CheckpointManager
from core.runtime import RuntimeManager
from core.types import TrainSummary
from models.transformer import PixelForCausalLM
from tokenizer.manager import PixelTokenizer
from training.data import TokenDataset, TokenDatasetConfig


def _seed_everything(seed: int) -> None:
    """Set Python and torch RNG state."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Create the AdamW optimizer."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
    )


def _build_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig) -> torch.optim.lr_scheduler.LambdaLR:
    """Create the warmup plus cosine-decay scheduler."""
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return float(step + 1) / float(max(config.warmup_steps, 1))
        progress = (step - config.warmup_steps) / float(max(config.total_steps - config.warmup_steps, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = config.min_learning_rate / config.learning_rate
        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a list of token windows into a batch."""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
    }


def _maybe_init_distributed(hardware) -> tuple[int, int]:
    """Initialize torch distributed when launched through torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if hardware.device == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
        if hardware.device == "cuda":
            torch.cuda.set_device(hardware.device_index)
    return rank, world_size


def train_model(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    tokenizer: PixelTokenizer,
    runtime_config: RuntimeConfig | None = None,
) -> TrainSummary:
    """Train a PIXEL model and return a compact summary."""
    runtime = RuntimeManager()
    hardware = runtime.detect_hardware(runtime_config, training_config)
    _seed_everything(training_config.seed)
    rank, world_size = _maybe_init_distributed(hardware)
    device = runtime.build_device(hardware)
    dataset = TokenDataset(TokenDatasetConfig(paths=(training_config.data_path,), sequence_length=training_config.sequence_length), tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=training_config.batch_size, sampler=sampler, shuffle=sampler is None, collate_fn=_collate)
    model = PixelForCausalLM(model_config).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[hardware.device_index] if hardware.device == "cuda" else None)
    optimizer = _build_optimizer(model, training_config)
    scheduler = _build_scheduler(optimizer, training_config)
    scaler = torch.amp.GradScaler(device="cuda", enabled=hardware.device == "cuda" and hardware.dtype == torch.float16)
    checkpoint_dir = Path(training_config.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    loss_history: list[float] = []
    iterator = iter(loader)
    model.train()
    for step in range(training_config.total_steps):
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(training_config.grad_accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            if hardware.device != "cpu":
                with torch.autocast(device_type=hardware.device, dtype=hardware.dtype):
                    output = model(input_ids)
                    loss = torch.nn.functional.cross_entropy(
                        output.logits.reshape(-1, model_config.vocab_size),
                        labels.reshape(-1),
                    )
                    loss = (loss + output.aux_loss) / training_config.grad_accumulation_steps
            else:
                output = model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    output.logits.reshape(-1, model_config.vocab_size),
                    labels.reshape(-1),
                )
                loss = (loss + output.aux_loss) / training_config.grad_accumulation_steps
            scaler.scale(loss).backward()
            accumulated += float(loss.item())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        loss_history.append(accumulated)
        if (step + 1) % training_config.save_every == 0 and rank == 0:
            checkpoint_manager.save(
                step=step + 1,
                model=model.module if isinstance(model, DistributedDataParallel) else model,
                optimizer=optimizer,
                scaler=scaler,
                config=training_config,
                metadata={"model": asdict(model_config), "hardware": hardware.to_dict()},
            )
    if rank == 0:
        checkpoint_manager.save(
            step=training_config.total_steps,
            model=model.module if isinstance(model, DistributedDataParallel) else model,
            optimizer=optimizer,
            scaler=scaler,
            config=training_config,
            metadata={"model": asdict(model_config), "hardware": hardware.to_dict()},
        )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return TrainSummary(
        output_dir=str(checkpoint_dir),
        steps_completed=training_config.total_steps,
        loss_history=loss_history[-10:],
        hardware=hardware.to_dict(),
    )
