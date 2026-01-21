# models/utils.py

from __future__ import annotations

from typing import Optional

import math
import torch
import torch.nn as nn


def trunc_normal_(tensor, mean=0., std=1., low=-2., high=2.):
    """Truncated normal initialization.

    This matches the behavior of timm/huggingface ViT implementations.
    """

    # Convert float bounds → tensor bounds
    low = (low - mean) / std
    high = (high - mean) / std

    # CDF for truncated distribution
    def norm_cdf(x):
        x = torch.tensor(x, dtype=tensor.dtype, device=tensor.device)
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    # Uniform sampling
    l = norm_cdf(low)
    h = norm_cdf(high)

    u = torch.empty_like(tensor).uniform_(l, h)

    # Inverse CDF
    tensor.copy_(
        mean + std * math.sqrt(2) * torch.erfinv(2 * u - 1)
    )

    # Clamp to ensure truncation limits
    tensor.clamp_(min=low * std + mean, max=high * std + mean)

    return tensor


def init_vit_weights(
    module: nn.Module,
    std: float = 0.02,
    bias_init: float = 0.0,
) -> None:
    """Initialize weights for Vision Transformer style modules.

    This can be used with `model.apply(init_vit_weights)`.

    Args:
        module: Module to initialize.
        std: Standard deviation for linear/embedding weights.
        bias_init: Bias initialization value.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        trunc_normal_(module.weight, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias, bias_init)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0.0)
        nn.init.constant_(module.weight, 1.0)
