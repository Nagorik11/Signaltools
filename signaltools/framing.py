"""Framing, detrending, and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .utils import DEFAULT_EPSILON, ensure_positive_int, to_1d_float_array

WindowType = Literal["rect", "hann", "hamming", "blackman"]


@dataclass
class FrameConfig:
    """Frame extraction parameters."""

    frame_size: int = 256
    hop_size: int = 128
    pad_end: bool = False
    window: WindowType = "rect"


def _window_values(size: int, window: WindowType) -> np.ndarray:
    if size <= 0:
        return np.array([], dtype=np.float64)
    if window == "rect":
        return np.ones(size, dtype=np.float64)
    if window == "hann":
        return np.hanning(size)
    if window == "hamming":
        return np.hamming(size)
    if window == "blackman":
        return np.blackman(size)
    raise ValueError(f"Unsupported window type: {window}")


def frame_signal(signal: list[float] | list[int], cfg: FrameConfig) -> list[list[float]]:
    """Split a signal into fixed-size frames, optionally padding the end."""
    ensure_positive_int(cfg.frame_size, "frame_size")
    ensure_positive_int(cfg.hop_size, "hop_size")
    s = to_1d_float_array(signal)
    n = len(s)
    if n == 0:
        return []

    window = _window_values(cfg.frame_size, cfg.window)
    frames: list[list[float]] = []
    last_start = max(0, n - cfg.frame_size)
    starts = list(range(0, max(0, n - cfg.frame_size + 1), cfg.hop_size))

    if cfg.pad_end and (not starts or starts[-1] != last_start):
        starts = list(range(0, n, cfg.hop_size))

    for start in starts:
        frame = s[start : start + cfg.frame_size]
        if len(frame) < cfg.frame_size:
            if not cfg.pad_end:  # pragma: no cover
                continue
            frame = np.pad(frame, (0, cfg.frame_size - len(frame)))
        frames.append((frame * window).astype(np.float64).tolist())
    return frames


def normalize_signal(signal: list[float] | list[int]) -> list[float]:
    """Normalize a signal by its peak amplitude."""
    s = to_1d_float_array(signal)
    if len(s) == 0:
        return []
    max_abs = float(np.max(np.abs(s)))
    return (s / max(max_abs, DEFAULT_EPSILON)).tolist()


def detrend_mean(signal: list[float] | list[int]) -> list[float]:
    """Subtract the mean from a signal."""
    s = to_1d_float_array(signal)
    if len(s) == 0:
        return []
    return (s - np.mean(s)).tolist()


def standardize_signal(signal: list[float] | list[int]) -> list[float]:
    """Standardize a signal to zero mean and unit variance."""
    s = to_1d_float_array(signal)
    if len(s) == 0:
        return []
    std = float(np.std(s))
    if std == 0.0:
        return [0.0 for _ in s]
    return ((s - np.mean(s)) / std).tolist()


__all__ = ["FrameConfig", "frame_signal", "normalize_signal", "detrend_mean", "standardize_signal"]
