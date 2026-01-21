# trainers/trainer.py

from __future__ import annotations

from typing import Any, Optional, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseModel
from trainers.abstract import AbstractTrainer
from trainers.callbacks.base import BaseCallback


class Trainer(AbstractTrainer):
    """Default trainer implementation following AbstractTrainer interface."""

    def __init__(
        self,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "auto",
        grad_clip: Optional[float] = None,
        callbacks: Optional[List[BaseCallback]] = None,
    ) -> None:
        """Initialize the Trainer."""

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.callbacks = callbacks or []

        # Device setup
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

    # ---------------------------------------------------------
    # Full training loop
    # ---------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
    ) -> None:
        """Run the full training loop."""

        # Callbacks: on_train_start
        for cb in self.callbacks:
            cb.on_train_start(self)

        for epoch in range(1, epochs + 1):
            for cb in self.callbacks:
                cb.on_epoch_start(self, epoch)

            train_logs = self.train_one_epoch(train_loader, epoch)

            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, train_logs)

            if val_loader is not None:
                val_logs = self.validate(val_loader, epoch)

                for cb in self.callbacks:
                    cb.on_validation_end(self, epoch, val_logs)

                if hasattr(self, "early_stopping") and self.early_stopping.should_stop:
                    break

            if self.scheduler is not None:
                self.scheduler.step()

        # Callbacks: on_train_end
        for cb in self.callbacks:
            cb.on_train_end(self)

    # ---------------------------------------------------------
    # Train one epoch
    # ---------------------------------------------------------
    def train_one_epoch(self, loader: DataLoader, epoch: int) -> dict[str, Any]:
        """Run a single training epoch."""
        self.model.train()
        logs = {}

        progress = tqdm(loader, desc=f"Train Epoch {epoch}")

        for step, batch in enumerate(progress):
            batch = self.model.to_device(batch)
            self.optimizer.zero_grad()

            out = self.model.training_step(batch, step)
            loss = out["loss"]
            other_logs = out.get("logs", {})

            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            logs = {"loss": loss.item(), **other_logs}
            progress.set_postfix(logs)

        return logs

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> dict[str, Any]:
        """Run validation."""
        self.model.eval()
        logs = {}

        progress = tqdm(loader, desc=f"Validation Epoch {epoch}")

        for step, batch in enumerate(progress):
            batch = self.model.to_device(batch)

            out = self.model.validation_step(batch, step)
            loss = out["loss"]
            other_logs = out.get("logs", {})

            logs = {"val_loss": loss.item(), **other_logs}
            progress.set_postfix(logs)

        return logs

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------
    @torch.no_grad()
    def predict(self, loader: DataLoader) -> List[Any]:
        """Run inference and return list of predictions."""
        self.model.eval()
        preds: List[Any] = []

        for step, batch in enumerate(loader):
            batch = self.model.to_device(batch)
            out = self.model.predict_step(batch, step)
            preds.append(out.cpu())

        return preds
