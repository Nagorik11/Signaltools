"""Logging utilities for signaltools."""

from __future__ import annotations

import logging
from typing import Final

DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def configure_logging(level: int | str = logging.INFO, fmt: str = DEFAULT_LOG_FORMAT) -> None:
    """Configure package-wide logging.

    Repeated calls are safe because `basicConfig(..., force=True)` rewrites the
    root configuration in a predictable way.
    """
    logging.basicConfig(level=level, format=fmt, force=True)


def get_logger(name: str = "signaltools") -> logging.Logger:
    """Return a namespaced logger for the package."""
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger", "DEFAULT_LOG_FORMAT"]
