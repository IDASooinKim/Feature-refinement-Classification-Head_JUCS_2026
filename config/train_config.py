# config/train_config.py

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Configuration for training loop."""

    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)

    batch_size: int = 32
    num_workers: int = 4

    grad_clip: float | None = 1.0

    # LR Scheduler
    scheduler: str | None = "cosine"  # 'step', 'linear', None
    warmup_epochs: int = 2

    # logging / checkpoints
    log_dir: str = "logs/"
    ckpt_dir: str = "checkpoints/"
