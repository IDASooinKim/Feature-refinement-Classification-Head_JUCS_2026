from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from .base import BaseModel


def build_adj_from_embeddings(
    X: torch.Tensor,
    k: int = 8,
    tau: float = 0.2,
    symmetrize: bool = True,
    add_self_loops: bool = True,
    clamp_neg_to_zero: bool = True,
) -> torch.Tensor:
    """
    X: (n, f) embeddings
    return A: (n, n) adjacency (dense)

    Steps:
    - cosine similarity S
    - optional clamp negatives
    - keep top-k neighbors per node
    - threshold by tau (optional)
    - symmetrize, add self loops
    """
    n, f = X.shape

    # cosine similarity (n,n)
    Xn = F.normalize(X, p=2, dim=1)
    S = Xn @ Xn.t()

    if clamp_neg_to_zero:
        S = S.clamp(min=0.0)

    # remove self similarity for neighbor selection
    S_no_diag = S.clone()
    S_no_diag.fill_diagonal_(-1e9)

    # top-k per row
    k = min(k, max(1, n - 1))
    vals, idx = torch.topk(S_no_diag, k=k, dim=1)

    A = torch.zeros_like(S)
    A.scatter_(1, idx, vals)

    # optional threshold
    if tau is not None:
        A = torch.where(A >= tau, A, torch.zeros_like(A))

    # symmetrize
    if symmetrize:
        A = torch.maximum(A, A.t())

    # add self loops
    if add_self_loops:
        A = A + torch.eye(n, device=A.device, dtype=A.dtype)

    return A


def gcn_normalize_adj(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    A: (n,n) adjacency (dense), assumed non-negative.
    return A_hat = D^{-1/2} A D^{-1/2}
    """
    deg = A.sum(dim=1)                      # (n,)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)   # (n,n)
    return D_inv_sqrt @ A @ D_inv_sqrt


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, use_bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # X: (n, f), A_hat: (n,n)
        X = self.dropout(X)
        return A_hat @ self.lin(X)
    

class GCN(nn.Module):
    """ResNet-based image classification model.

    Args:
        backbone: ResNet variant name. One of ["resnet18", "resnet34", "resnet50"].
        in_chans: Number of input channels.
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.
    """

class ResNetClassifier(BaseModel):
    """ResNet-based image classification model.

    Args:
        backbone: ResNet variant name. One of ["resnet18", "resnet34", "resnet50"].
        in_chans: Number of input channels.
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        in_chans: int = 3,
        num_classes: int = 1000,
        pretrained: bool = False,
        k: int = 1,
        tau: float = 0.2
    ) -> None:
        super().__init__()

        # ---------------------------
        # Initialize
        # ---------------------------
        self.k = k
        self.tau = tau

        # ---------------------------
        # Build backbone
        # ---------------------------
        if backbone == "resnet18":
            self.backbone = tv_models.resnet18(pretrained=pretrained)
        elif backbone == "resnet34":
            self.backbone = tv_models.resnet34(pretrained=pretrained)
        elif backbone == "resnet50":
            self.backbone = tv_models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ---------------------------
        # Adjust input channels if needed
        # ---------------------------
        if in_chans != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_chans,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

        # ---------------------------
        # Replace classification head
        # ---------------------------
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 512)
        
        # ---------------------------
        # Call GCN Layer
        # ---------------------------
        self.gcn = GCNLayer(in_features, in_features)
        self.linear1 = nn.Linear(in_features, in_features)
        self.linear2 = nn.Linear(in_features, num_classes)
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Initialize weights
        self.setup()

    # ---------------------------------------------------------
    # Setup / initialization
    # ---------------------------------------------------------
    def setup(self) -> None:
        """Initialize classification head weights."""
        with torch.no_grad():
            if isinstance(self.backbone.fc, nn.Linear):
                nn.init.normal_(self.backbone.fc.weight, mean=0.0, std=0.02)
                if self.backbone.fc.bias is not None:
                    nn.init.zeros_(self.backbone.fc.bias)

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
        x = self.backbone(x)

        A = build_adj_from_embeddings(x, k=self.k, tau=self.tau)
        A_hat = gcn_normalize_adj(A)

        x = self.gcn(x, A_hat)
        x = self.linear1(x)
        logits = self.linear2(x)
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
