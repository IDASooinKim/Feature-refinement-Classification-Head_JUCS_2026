# trainers/abstract.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from torch.utils.data import DataLoader


class AbstractTrainer(ABC):
    """Base interface for all trainers.

    Defines the minimal set of methods that every trainer implementation
    must provide.
    """

    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
    ) -> None:
        """Run full training including optional validation.

        Args:
            train_loader: DataLoader for training dataset.
            val_loader: Optional validation DataLoader.
            epochs: Number of epochs to train.
        """
        pass

    @abstractmethod
    def train_one_epoch(self, loader: DataLoader, epoch: int) -> None:
        """Run a single training epoch.

        Args:
            loader: Training DataLoader.
            epoch: Current epoch index (1-based).
        """
        pass

    @abstractmethod
    def validate(self, loader: DataLoader, epoch: int) -> None:
        """Run validation over the provided DataLoader.

        Args:
            loader: Validation DataLoader.
            epoch: Current epoch index.
        """
        pass

    @abstractmethod
    def predict(self, loader: DataLoader) -> list[Any]:
        """Run inference over a DataLoader.

        Args:
            loader: DataLoader for inference.

        Returns:
            List of prediction outputs.
        """
        pass
