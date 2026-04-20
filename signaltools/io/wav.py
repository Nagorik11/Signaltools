"""Backward-compatible WAV helpers.

Canonical implementations live in `signaltools.io`.
"""

from __future__ import annotations

import numpy as np

from . import read_wav, write_wav


def save_wav(data: np.ndarray, filename: str, sample_rate: int = 44100) -> None:
    """Save a normalized numpy array as WAV."""
    write_wav(filename, data.tolist(), sample_rate=sample_rate)


def load_wav(filename: str) -> np.ndarray:
    """Load a WAV file into a normalized numpy array."""
    return np.asarray(read_wav(filename), dtype=np.float32)


__all__ = ["save_wav", "load_wav"]
