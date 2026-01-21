# trainers/callbacks/early_stopping.py

from __future__ import annotations
from typing import Any, Optional

from .base import BaseCallback


class EarlyStoppingCallback(BaseCallback):
    """Stop training early when a monitored metric fails to improve.

    Args:
        monitor: Metric to monitor (e.g., "val/loss").
        mode: "min" or "max".
        patience: Allowed epochs without improvement.
    """

    def __init__(self, monitor: str = "val/loss", mode: str = "min", patience: int = 5) -> None:
        assert mode in ["min", "max"], "mode must be 'min' or 'max'"

        self.monitor = monitor
        self.mode = mode
        self.patience = patience

        self.best: Optional[float] = None
        self.wait = 0
        self.should_stop = False

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best
        return current > best

    def on_validation_end(self, trainer: Any, epoch: int, logs: dict[str, Any]) -> None:
        value = logs.get(self.monitor)

        if value is None:
            print(f"[EarlyStopping] Warning: metric '{self.monitor}' not found in logs.")
            return

        if self.best is None:
            self.best = value
            return

        if self._is_better(value, self.best):
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                print(f"[EarlyStopping] No improvement for {self.patience} epochs — stopping.")
