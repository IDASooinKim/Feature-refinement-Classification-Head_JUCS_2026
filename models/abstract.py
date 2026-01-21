# models/abstract.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class AbstractModel(nn.Module, ABC):
    """Base interface for all models.

    Models must implement:
        - forward()
        - training_step()
        - validation_step()
        - predict_step()
    """

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Compute the forward pass.

        Args:
            x: Arbitrary input (tensor, dict, tuple, etc.).

        Returns:
            The model output.
        """
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch: Any, step: int) -> dict[str, Any]:
        """Run a single training step.

        This should:
            - call forward()
            - compute loss
            - optionally compute metrics

        Args:
            batch: A training batch.
            step: Global step index.

        Returns:
            A dictionary containing:
                "loss": torch.Tensor
                "logs": dict[str, Any]
        """
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch: Any, step: int) -> dict[str, Any]:
        """Run a single validation step.

        Args:
            batch: A validation batch.
            step: Global step index.

        Returns:
            A dictionary containing:
                "loss": torch.Tensor
                "logs": dict[str, Any]
        """
        raise NotImplementedError

    @abstractmethod
    def predict_step(self, batch: Any, step: int) -> Any:
        """Run a single inference step.

        Args:
            batch: Input batch for inference.
            step: Step index.

        Returns:
            Model predictions.
        """
        raise NotImplementedError
