"""Shared utilities for logging, reproducibility, and safe file operations."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np


def configure_logging(log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure a single project logger with console and file handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hospital_repair")
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def set_random_seed(seed: int) -> None:
    """Set pseudo-random seeds used by Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Persist a JSON serializable payload using readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def occupancy_class(rate: float) -> str:
    """Convert continuous occupancy into an operational risk class."""
    if rate >= 85:
        return "High"
    if rate >= 70:
        return "Moderate"
    return "Low"
