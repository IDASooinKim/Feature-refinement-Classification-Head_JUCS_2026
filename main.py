# main.py

from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import ModelConfig, TrainConfig, DataConfig
from models.resnet import ResNetClassifier
from trainers import Trainer, LoggingCallback, EarlyStoppingCallback, ModelCheckpointCallback
from utils import get_logger, init_distributed_mode
from data.preprocess import (
    load_samples_from_folder,
    train_val_split,
    build_datasets,
    build_dataloaders,
)
from data.collator import MixupCutmixCollator, DefaultCollator


def main() -> None:
    # ---------------------------------------------------------
    # Initialize distributed training (DDP-friendly)
    # ---------------------------------------------------------
    init_distributed_mode()

    # ---------------------------------------------------------
    # Load configs
    # ---------------------------------------------------------
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    data_cfg = DataConfig()

    # ---------------------------------------------------------
    # Logger
    # ---------------------------------------------------------
    logger = get_logger("train", log_file=f"{train_cfg.log_dir}/train.log")
    logger.info("Starting training...")

    # ---------------------------------------------------------
    # Dataset preparation
    # ---------------------------------------------------------
    samples = load_samples_from_folder(data_cfg.dataset_root)
    train_samples, val_samples = train_val_split(samples, data_cfg.val_ratio)

    train_ds, val_ds = build_datasets(
        train_samples,
        val_samples,
        img_size=data_cfg.img_size,
    )

    # Collator (mixup/cutmix optional)
    if data_cfg.use_mixup or data_cfg.use_cutmix:
        collator = MixupCutmixCollator(
            mixup_alpha=data_cfg.mixup_alpha if data_cfg.use_mixup else 0.0,
            cutmix_alpha=data_cfg.cutmix_alpha if data_cfg.use_cutmix else 0.0,
            num_classes=model_cfg.num_classes,
        )
    else:
        collator = DefaultCollator()

    train_loader, val_loader = build_dataloaders(
        train_ds,
        val_ds,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        collator=collator,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = ResNetClassifier(
        backbone="resnet18",
        in_chans=3,
        num_classes=10,
        pretrained=True
    )

    # ---------------------------------------------------------
    # Optimizer & Scheduler
    # ---------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=train_cfg.betas,
    )

    scheduler = None
    if train_cfg.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=train_cfg.epochs,
        )

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------
    callbacks = [
        LoggingCallback(),
        EarlyStoppingCallback(monitor="val_loss", patience=3),
        ModelCheckpointCallback(
            dirpath=train_cfg.ckpt_dir,
            monitor="val_loss",
            mode="min",
            filename="best_{score:.4f}.pt",
            save_best_only=True,
        ),
    ]

    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_clip=train_cfg.grad_clip,
        callbacks=callbacks,
        device="auto",
    )

    # ---------------------------------------------------------
    # Run training
    # ---------------------------------------------------------
    logger.info("Start fitting model...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg.epochs,
    )

    logger.info("Training complete.")


if __name__ == "__main__":

    main()
