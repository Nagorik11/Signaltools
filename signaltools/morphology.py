"""Morphological and order-statistic filters for 1D signals."""

from __future__ import annotations

import numpy as np

from .utils import ensure_positive_int, to_1d_float_array


def _validate_window(window_size: int) -> int:
    window_size = ensure_positive_int(window_size, "window_size")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    return window_size


def _ordered_window_filter(signal: list[float] | list[int], window_size: int, rank: int, pad_mode: str = "edge") -> list[float]:
    x = to_1d_float_array(signal)
    if x.size == 0:
        return []
    window_size = _validate_window(window_size)
    if rank < 0 or rank >= window_size:
        raise ValueError("rank must be between 0 and window_size - 1")
    half = window_size // 2
    padded = np.pad(x, (half, half), mode=pad_mode)
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = np.sort(padded[i : i + window_size])[rank]
    return out.tolist()


def advanced_median_filter(signal: list[float] | list[int], window_size: int = 5, pad_mode: str = "edge") -> list[float]:
    window_size = _validate_window(window_size)
    return _ordered_window_filter(signal, window_size, window_size // 2, pad_mode=pad_mode)


def rank_filter(signal: list[float] | list[int], window_size: int = 5, rank: int = 0, pad_mode: str = "edge") -> list[float]:
    return _ordered_window_filter(signal, window_size, rank, pad_mode=pad_mode)


def dilation_1d(signal: list[float] | list[int], window_size: int = 3, pad_mode: str = "edge") -> list[float]:
    return rank_filter(signal, window_size=window_size, rank=_validate_window(window_size) - 1, pad_mode=pad_mode)


def erosion_1d(signal: list[float] | list[int], window_size: int = 3, pad_mode: str = "edge") -> list[float]:
    return rank_filter(signal, window_size=window_size, rank=0, pad_mode=pad_mode)


def opening_1d(signal: list[float] | list[int], window_size: int = 3, pad_mode: str = "edge") -> list[float]:
    return dilation_1d(erosion_1d(signal, window_size, pad_mode), window_size, pad_mode)


def closing_1d(signal: list[float] | list[int], window_size: int = 3, pad_mode: str = "edge") -> list[float]:
    return erosion_1d(dilation_1d(signal, window_size, pad_mode), window_size, pad_mode)


def morphological_gradient_1d(signal: list[float] | list[int], window_size: int = 3, pad_mode: str = "edge") -> list[float]:
    dil = np.asarray(dilation_1d(signal, window_size, pad_mode), dtype=np.float64)
    ero = np.asarray(erosion_1d(signal, window_size, pad_mode), dtype=np.float64)
    return (dil - ero).tolist()


__all__ = [
    "advanced_median_filter",
    "rank_filter",
    "dilation_1d",
    "erosion_1d",
    "opening_1d",
    "closing_1d",
    "morphological_gradient_1d",
]
