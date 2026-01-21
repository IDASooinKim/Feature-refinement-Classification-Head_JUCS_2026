# config/data_config.py

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration for dataset and dataloader."""

    dataset_root: str = r"C:\Users\fromf\Desktop\codes\torch_convention-main\torch_convention-main\dataset"
    val_ratio: float = 0.2

    img_size: int = 224

    # Collator
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0
