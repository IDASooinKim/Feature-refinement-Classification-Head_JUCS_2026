# data/dataset.py

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import os

from PIL import Image
import torch
from torch.utils.data import Dataset


class ImageClassificationDataset(Dataset):
    """Generic dataset for supervised image classification.

    Supports:
        - A list of (image_path, label) tuples
        - Optional torchvision-style transforms
        - Lazy image loading using PIL

    Examples:
        samples = [
            ("/path/to/img1.jpg", 0),
            ("/path/to/img2.jpg", 1),
        ]
        dataset = ImageClassificationDataset(samples, transform=transform)

    Args:
        samples: A list of tuples, each containing
            (image_path: str, label: int).
        transform: Optional callable applied to the loaded PIL image.
        return_index: If True, includes `"index"` in each returned sample.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[Callable] = None,
        return_index: bool = False,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.return_index = return_index

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Retrieve a single sample.

        Args:
            index: Index of the desired sample.

        Returns:
            A dictionary containing:
                - "image": Tensor(C, H, W) after transform
                - "label": Tensor(dtype=torch.long)
                - Optional "index": int
        """
        path, label = self.samples[index]

        if not os.path.exists(path):
            raise FileNotFoundError(f"[ImageClassificationDataset] Image not found: {path}")

        # Load image
        image = Image.open(path).convert("RGB")

        # Apply transform (e.g., Resize, ToTensor, Normalize)
        if self.transform is not None:
            image = self.transform(image)

        output: Dict[str, Any] = {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }

        if self.return_index:
            output["index"] = index

        return output
