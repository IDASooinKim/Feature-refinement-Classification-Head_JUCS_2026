# trainers/callbacks/base.py

from __future__ import annotations
from typing import Any


class BaseCallback:
    """Base interface for callbacks used in Trainer.

    All callback events are optional. Subclasses may implement:
        - on_train_start
        - on_train_end
        - on_epoch_start
        - on_epoch_end
        - on_validation_end
    """

    def on_train_start(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, Any]) -> None:
        pass

    def on_validation_end(self, trainer: Any, epoch: int, logs: dict[str, Any]) -> None:
        pass
