# trainers/callbacks/logging.py

from __future__ import annotations
from typing import Any

from .base import BaseCallback


class LoggingCallback(BaseCallback):
    """Print logs collected during training and validation."""

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, Any]) -> None:
        print(f"[Epoch {epoch}] Training Metrics: {logs}")

    def on_validation_end(self, trainer: Any, epoch: int, logs: dict[str, Any]) -> None:
        print(f"[Epoch {epoch}] Validation Metrics: {logs}")
