# config/model_config.py

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for Vision Transformer model."""

    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_classes: int = 85742
    dropout: float = 0.1
    attention_dropout: float = 0.1
