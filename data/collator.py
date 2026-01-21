# data/collator.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class DefaultCollator:
    """Default collator for image classification.

    Stacks images and labels into batch tensors.

    Args:
        keep_index: Whether to include the 'index' field when present in samples.
    """

    def __init__(self, keep_index: bool = False) -> None:
        self.keep_index = keep_index

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch of samples into tensors.

        Args:
            batch: A list of dataset samples. Each sample must contain:
                - 'image': Tensor(C, H, W)
                - 'label': Tensor(...)
                - Optional 'index': int

        Returns:
            A dict with:
                - 'image': Tensor(B, C, H, W)
                - 'label': Tensor(B, ...)
                - Optional 'index': Tensor(B,)
        """
        images = torch.stack([item["image"] for item in batch], dim=0)
        labels = torch.stack([item["label"] for item in batch], dim=0)

        output: Dict[str, Any] = {
            "image": images,
            "label": labels,
        }

        if self.keep_index and "index" in batch[0]:
            indices = torch.tensor([item["index"] for item in batch], dtype=torch.long)
            output["index"] = indices

        return output


# ---------------------------------------------------------
# Mixup & Cutmix Collator
# ---------------------------------------------------------

class MixupCutmixCollator(DefaultCollator):
    """Collator with optional Mixup and Cutmix augmentation.

    If mixup_alpha > 0, mixup is applied.
    If cutmix_alpha > 0, cutmix is applied.
    If both > 0, mixup takes priority.

    Args:
        keep_index: Whether to return sample indices.
        mixup_alpha: Mixup alpha parameter for Beta distribution.
        cutmix_alpha: Cutmix alpha parameter for Beta distribution.
        prob: Probability of applying augmentation per batch.
        num_classes: Required for generating one-hot labels.
    """

    def __init__(
        self,
        keep_index: bool = False,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        prob: float = 1.0,
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__(keep_index=keep_index)

        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes

    def _rand_bbox(self, width: int, height: int, lam: float) -> Tuple[int, int, int, int]:
        """Generate a random bounding box for CutMix.

        Args:
            width: Image width.
            height: Image height.
            lam: Lambda value sampled from Beta distribution.

        Returns:
            (x0, y0, x1, y1): Coordinates of the CutMix bounding box.
        """
        cut_w = int(width * (1 - lam) ** 0.5)
        cut_h = int(height * (1 - lam) ** 0.5)

        cx = torch.randint(0, width, (1,)).item()
        cy = torch.randint(0, height, (1,)).item()

        x0 = max(cx - cut_w // 2, 0)
        x1 = min(cx + cut_w // 2, width)
        y0 = max(cy - cut_h // 2, 0)
        y1 = min(cy + cut_h // 2, height)

        return x0, y0, x1, y1

    def _one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert class indices into one-hot vectors."""
        if self.num_classes is None:
            raise ValueError("num_classes must be set for Mixup/Cutmix one-hot labels.")

        return F.one_hot(labels, num_classes=self.num_classes).float()

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply Mixup or Cutmix to a collated batch.

        Args:
            batch: A list of dataset samples.

        Returns:
            A dict containing augmented batch tensors.
        """
        output = super().__call__(batch)
        images, labels = output["image"], output["label"]

        # Decide whether to apply augmentation
        if torch.rand(1).item() > self.prob:
            return output

        B, C, H, W = images.shape
        index = torch.randperm(B)

        # Select augmentation type
        use_mixup = self.mixup_alpha > 0
        use_cutmix = self.cutmix_alpha > 0

        if not (use_mixup or use_cutmix):
            return output

        # Mixup
        if use_mixup:
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
            images = lam * images + (1 - lam) * images[index]
        else:
            # CutMix
            lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample().item()
            x0, y0, x1, y1 = self._rand_bbox(W, H, lam)
            images[:, :, y0:y1, x0:x1] = images[index, :, y0:y1, x0:x1]

            # Adjust lambda based on area replaced
            lam = 1 - ((x1 - x0) * (y1 - y0) / (W * H))

        y1 = self._one_hot(labels)
        y2 = self._one_hot(labels[index])
        labels = lam * y1 + (1 - lam) * y2

        output["image"] = images
        output["label"] = labels
        return output
