# models/base.py

from __future__ import annotations
from typing import Any, Union, Optional

import torch
import torch.nn as nn

from .abstract import AbstractModel


class BaseModel(AbstractModel):
    """Base class implementing common model utilities."""

    # ---------------------------------------------------------
    # Optional setup hook
    # ---------------------------------------------------------
    def setup(self) -> None:
        """Optional setup hook for weight initialization or other logic."""
        return

    # ---------------------------------------------------------
    # Device utilities
    # ---------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def to_device(self, batch: Any) -> Any:
        """Recursively move batch data to model device."""
        if torch.is_tensor(batch):
            return batch.to(self.device)

        if isinstance(batch, dict):
            return {k: self.to_device(v) for k, v in batch.items()}

        if isinstance(batch, (list, tuple)):
            return type(batch)(self.to_device(x) for x in batch)

        return batch  # non-tensor data

    # ---------------------------------------------------------
    # Checkpoint utilities
    # ---------------------------------------------------------
    def save(
        self,
        path: str,
        *,
        save_dtype: Optional[torch.dtype] = None,
        _use_new_zipfile_serialization: bool = True,
    ) -> None:
        """Save the model's state dict.

        Args:
            path: File path.
            save_dtype: Optionally convert state dict dtype before saving.
            _use_new_zipfile_serialization: torch.save backend flag.
        """
        state_dict = self.state_dict()

        # Optionally convert dtype (e.g., float32 → float16)
        if save_dtype is not None:
            state_dict = {
                k: v.to(save_dtype) if torch.is_tensor(v) else v
                for k, v in state_dict.items()
            }

        torch.save(
            state_dict,
            path,
            _use_new_zipfile_serialization=_use_new_zipfile_serialization,
        )

    def load(
        self,
        path: str,
        *,
        map_location: Optional[Union[str, torch.device]] = "cpu",
        strict: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Load model weights.

        Args:
            path: File path.
            map_location: Where to load the model (e.g., "cpu" or "cuda").
            strict: Whether to strictly enforce matching keys.
            dtype: Optionally cast weights after loading.
        """

        state = torch.load(path, map_location=map_location)

        if dtype is not None:
            state = {
                k: v.to(dtype) if torch.is_tensor(v) else v
                for k, v in state.items()
            }

        self.load_state_dict(state, strict=strict)

    # ---------------------------------------------------------
    # Parameter info
    # ---------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}\nTrainable params: {self.count_parameters():,}"
