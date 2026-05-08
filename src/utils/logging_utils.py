"""Shared logging configuration."""

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s — %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    return logger
