# trainers/callbacks/checkpoint.py

from __future__ import annotations
from typing import Any, Optional
import os

from .base import BaseCallback


class ModelCheckpointCallback(BaseCallback):
    """Save model checkpoints based on a monitored metric.

    Args:
        dirpath: Directory to save checkpoint files.
        monitor: Metric to monitor (e.g., "val/loss").
        mode: "min" or "max".
        filename: Format string for saving.
        save_best_only: If True, only save when metric improves.
    """

    def __init__(
        self,
        dirpath: str = "checkpoints",
        monitor: str = "val/loss",
        mode: str = "min",
        filename: str = "epoch_{epoch}_score_{score:.4f}.pt",
        save_best_only: bool = True,
    ) -> None:
        os.makedirs(dirpath, exist_ok=True)

        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        self.save_best_only = save_best_only

        self.best: Optional[float] = None

        assert mode in ["min", "max"], "mode must be 'min' or 'max'"

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best
        return current > best

    def on_validation_end(self, trainer: Any, epoch: int, logs: dict[str, Any]) -> None:
        value = logs.get(self.monitor)

        if value is None:
            print(f"[Checkpoint] Warning: metric '{self.monitor}' not found in logs.")
            return

        # Determine save path
        save_path = os.path.join(
            self.dirpath,
            self.filename.format(epoch=epoch, score=value),
        )

        # Save only if better
        if self.save_best_only:
            if self.best is None or self._is_better(value, self.best):
                self.best = value
                trainer.model.save(save_path)
                print(f"[Checkpoint] Saved BEST model to {save_path}")
        else:
            trainer.model.save(save_path)
            print(f"[Checkpoint] Saved model to {save_path}")
