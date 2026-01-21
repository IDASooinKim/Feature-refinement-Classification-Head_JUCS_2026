# models/transformer.py

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .base import BaseModel
from ._layers import PatchEmbedding, TransformerBlock
from .utils import trunc_normal_


class VisionTransformer(BaseModel):
    """Vision Transformer model for image classification.

    Args:
        img_size: Input image size (assumed square).
        patch_size: Patch size.
        in_chans: Number of input channels.
        num_classes: Output classes for classification.
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio for hidden size in MLP block.
        dropout: Dropout probability applied after positional embedding.
        attention_dropout: Dropout used inside attention module.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,  # ★ main.py와 이름 통일
    ) -> None:
        super().__init__()

        # ---------------------------
        # Patch embedding
        # ---------------------------
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # ---------------------------
        # Class token & positional embedding
        # ---------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        self.pos_drop = nn.Dropout(dropout)

        # ---------------------------
        # Transformer blocks
        # ---------------------------
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attention_dropout,  # ★ 이름 통일
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # ---------------------------
        # Classification head
        # ---------------------------
        self.head = nn.Linear(embed_dim, num_classes)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Initialize weights
        self.setup()

    # ---------------------------------------------------------
    # Setup / initialization
    # ---------------------------------------------------------
    def setup(self) -> None:
        """Initialize model weights."""
        with torch.no_grad():
            trunc_normal_(self.cls_token, std=0.02)
            trunc_normal_(self.pos_embed, std=0.02)

            if isinstance(self.head, nn.Linear):
                trunc_normal_(self.head.weight, std=0.02)
                if self.head.bias is not None:
                    nn.init.zeros_(self.head.bias)


    # ---------------------------------------------------------
    # Core forward
    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, C, H, W)

        Returns:
            Logits with shape (B, num_classes)
        """
        B = x.shape[0]

        x = self.patch_embed(x)  # (B, N, D)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + N, D)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # CLS output
        cls_output = x[:, 0]  # (B, D)
        logits = self.head(cls_output)
        return logits

    # ---------------------------------------------------------
    # Training / validation / prediction steps
    # ---------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Single training step."""
        batch = self.to_device(batch)
        images = batch["image"]
        labels = batch["label"]

        logits = self(images)
        loss = self.criterion(logits, labels)

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()

        logs = {
            "train/loss": float(loss.item()),
            "train/acc": acc,
        }
        return {"loss": loss, "logs": logs}

    def validation_step(self, batch: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Single validation step."""
        batch = self.to_device(batch)
        images = batch["image"]
        labels = batch["label"]

        logits = self(images)
        loss = self.criterion(logits, labels)

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()

        logs = {
            "val/loss": float(loss.item()),
            "val/acc": acc,
        }
        return {"loss": loss, "logs": logs}

    def predict_step(self, batch: Dict[str, Any], step: int) -> torch.Tensor:
        """Prediction step."""
        batch = self.to_device(batch)
        images = batch["image"]
        return self(images)
