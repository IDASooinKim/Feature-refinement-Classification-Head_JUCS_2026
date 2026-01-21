# data/preprocess.py

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import os
import glob
import random

from torch.utils.data import DataLoader

from torchvision import transforms

from .dataset import ImageClassificationDataset
from .collator import DefaultCollator


# ---------------------------------------------------------
# Sample loading
# ---------------------------------------------------------
def load_samples_from_folder(root: str) -> List[Tuple[str, int]]:
    """Load image paths and labels assuming folder/class_name/file.jpg structure.

    Example directory:
        root/
            cat/
                0001.jpg
                0002.jpg
            dog/
                0010.jpg

    Args:
        root: Root directory containing class subfolders.

    Returns:
        A list of (image_path, class_id) samples.
    """
    samples: List[Tuple[str, int]] = []
    class_names = sorted(os.listdir(root))

    label_map = {cls: idx for idx, cls in enumerate(class_names)}

    for cls in class_names:
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue

        for path in glob.glob(os.path.join(cls_dir, "*")):
            if path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                samples.append((path, label_map[cls]))

    return samples


# ---------------------------------------------------------
# Train/val split
# ---------------------------------------------------------
def train_val_split(
    samples: List[Tuple[str, int]],
    val_ratio: float = 0.2,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Split samples into train and validation sets.

    Args:
        samples: List of (path, label) samples.
        val_ratio: Fraction of samples used for validation.
        shuffle: Whether to shuffle before splitting.
        seed: RNG seed for reproducible shuffling.

    Returns:
        train_samples, val_samples
    """
    if shuffle:
        random.seed(seed)
        random.shuffle(samples)

    n_val = int(len(samples) * val_ratio)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    return train_samples, val_samples


# ---------------------------------------------------------
# Build transforms
# ---------------------------------------------------------
def build_transforms(
    img_size: int = 224,
    is_train: bool = True,
) -> Any:
    """Create torchvision transforms for training or validation.

    Args:
        img_size: Output image size (square).
        is_train: Whether to build training augmentation.

    Returns:
        transform: A callable transform pipeline.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


# ---------------------------------------------------------
# Build datasets
# ---------------------------------------------------------
def build_datasets(
    train_samples: List[Tuple[str, int]],
    val_samples: Optional[List[Tuple[str, int]]],
    img_size: int = 224,
) -> Tuple[ImageClassificationDataset, Optional[ImageClassificationDataset]]:
    """Create training and validation datasets.

    Args:
        train_samples: List of (path, label) for training.
        val_samples: Validation samples (optional).
        img_size: Target image size.

    Returns:
        train_dataset, val_dataset
    """
    train_transform = build_transforms(img_size, is_train=True)
    val_transform = build_transforms(img_size, is_train=False)

    train_dataset = ImageClassificationDataset(
        samples=train_samples,
        transform=train_transform,
    )

    val_dataset = None
    if val_samples is not None:
        val_dataset = ImageClassificationDataset(
            samples=val_samples,
            transform=val_transform,
        )

    return train_dataset, val_dataset


# ---------------------------------------------------------
# Build dataloaders
# ---------------------------------------------------------
def build_dataloaders(
    train_dataset: ImageClassificationDataset,
    val_dataset: Optional[ImageClassificationDataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    collator: Optional[Any] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create DataLoader objects for training and validation.

    Args:
        train_dataset: Dataset for training.
        val_dataset: Optional validation dataset.
        batch_size: Training batch size.
        num_workers: Dataloader workers.
        collator: Optional collate_fn. If None, DefaultCollator is used.

    Returns:
        train_loader, val_loader
    """
    if collator is None:
        collator = DefaultCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
        )

    return train_loader, val_loader
