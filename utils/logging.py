# utils/logging.py

from __future__ import annotations

import logging
import os
from typing import Optional

from .distributed import is_main_process


def get_logger(
    name: str = "app",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create and configure a logger.

    Features:
        - Main process only logging (for distributed training)
        - Optional file logging
        - Pretty, concise formatting

    Args:
        name: Logger name.
        log_file: Optional file path for saving logs.
        level: Logging level (e.g., logging.INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ---------------------------
    # Console Handler (rank 0 only)
    # ---------------------------
    if is_main_process():
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(ch)

    # ---------------------------
    # File Handler (optional)
    # ---------------------------
    if log_file is not None and is_main_process():
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    return logger


def disable_external_loggers() -> None:
    """Disable noisy external loggers such as PIL, matplotlib, etc."""
    for noisy in ["PIL", "matplotlib", "torchvision"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
