# config/__init__.py

from .model_config import ModelConfig
from .train_config import TrainConfig
from .data_config import DataConfig

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "DataConfig",
]
