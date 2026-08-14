"""Logging setup — single logger instance shared across the pipeline."""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path


def setup_logger(
    name: str = "ksp_neuro",
    level: int | str = logging.INFO,
    log_dir: str = "./logs",
) -> logging.Logger:
    """Create (or retrieve) a named logger with console + file handlers.

    Parameters
    ----------
    name : str
        Logger namespace prefix  e.g. ``ksp_neuro.ne_engine``.
    level : int | str
        Logging level constant or string name.
    log_dir : str
        Directory for the rotating file handler (auto-created).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file handler (rotating, 10 MB max)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        log_path / f"{name.replace('.', '_')}.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


__all__ = ["setup_logger"]
