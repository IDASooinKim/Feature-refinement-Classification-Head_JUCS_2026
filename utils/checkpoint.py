# utils/checkpoint.py

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from .distributed import is_main_process


# ---------------------------------------------------------
# Basic model save / load
# ---------------------------------------------------------
def save_checkpoint(
    path: str,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    scheduler_state: Optional[Dict[str, Any]] = None,
    **metadata: Any,
) -> None:
    """Save a checkpoint containing model weights + optimizer + scheduler + metadata.

    Args:
        path: File path to store checkpoint.
        model_state: State dict of the model.
        optimizer_state: Optional optimizer state dict.
        scheduler_state: Optional scheduler state dict.
        metadata: Additional training info (epoch, metrics, etc.).
    """
    if not is_main_process():
        return  # Only rank 0 saves checkpoints

    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload = {
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "metadata": metadata,
    }

    torch.save(payload, path)
    print(f"[checkpoint] Saved checkpoint: {path}")


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load checkpoint from disk.

    Args:
        path: Path to checkpoint file.
        map_location: Device to map checkpoint tensors to.

    Returns:
        The loaded checkpoint dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[checkpoint] File not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)
    print(f"[checkpoint] Loaded checkpoint: {path}")
    return checkpoint


# ---------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------
def save_last(
    dirpath: str,
    model: Any,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    **metrics: Any,
) -> str:
    """Save the latest checkpoint.

    Args:
        dirpath: Directory to store checkpoint.
        model: PyTorch model (must have .state_dict()).
        optimizer: Optimizer for optional saving.
        scheduler: Scheduler for optional saving.
        epoch: Current epoch number.
        metrics: Additional metrics to store.

    Returns:
        Path to the saved checkpoint.
    """
    path = os.path.join(dirpath, "last.pt")

    save_checkpoint(
        path,
        model.state_dict(),
        optimizer.state_dict() if optimizer else None,
        scheduler.state_dict() if scheduler else None,
        epoch=epoch,
        **metrics,
    )

    return path


def save_best(
    dirpath: str,
    model: Any,
    best_metric: float,
    metric_name: str = "val_loss",
    **extra: Any,
) -> str:
    """Save a checkpoint when the metric improves.

    Args:
        dirpath: Directory to store checkpoint.
        model: Model to save.
        best_metric: Metric value to store.
        metric_name: Tag for the metric.
        extra: Additional metadata.

    Returns:
        Path to the saved best checkpoint.
    """
    filename = f"best_{metric_name}.pt"
    path = os.path.join(dirpath, filename)

    save_checkpoint(
        path,
        model.state_dict(),
        best_metric=best_metric,
        metric_name=metric_name,
        **extra,
    )
    return path
