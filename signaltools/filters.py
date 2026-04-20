"""Filtering utilities for 1D signals."""

from __future__ import annotations

import numpy as np

from .utils import DEFAULT_EPSILON, ensure_non_negative_float, ensure_positive_int, to_1d_float_array


def moving_average(signal: list[float] | list[int], window_size: int = 5) -> list[float]:
    """Smooth a signal with a centered moving average."""
    w = ensure_positive_int(window_size, "window_size")
    s = to_1d_float_array(signal)
    if len(s) == 0:
        return []
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(s, kernel, mode="same").astype(np.float64).tolist()


def median_filter(signal: list[float] | list[int], window_size: int = 5) -> list[float]:
    """Apply a median filter using edge padding."""
    w = ensure_positive_int(window_size, "window_size")
    if w % 2 == 0:
        w += 1
    s = to_1d_float_array(signal)
    if len(s) == 0:
        return []
    pad = w // 2
    padded = np.pad(s, (pad, pad), mode="edge")
    return [float(np.median(padded[i : i + w])) for i in range(len(s))]


def remove_dc(signal: list[float] | list[int]) -> list[float]:
    """Remove the DC component by subtracting the mean."""
    s = to_1d_float_array(signal)
    if len(s) == 0:
        return []
    return (s - np.mean(s)).astype(np.float64).tolist()


def normalize_peak(signal: list[float] | list[int]) -> list[float]:
    """Normalize a signal by its peak absolute amplitude."""
    s = to_1d_float_array(signal)
    if len(s) == 0:
        return []
    peak = float(np.max(np.abs(s)))
    return (s / max(peak, DEFAULT_EPSILON)).astype(np.float64).tolist()


def fft_bandpass(signal: list[float] | list[int], sample_rate: int, low_hz: float, high_hz: float) -> list[float]:
    """Filter a signal by zeroing FFT bins outside the requested passband."""
    sample_rate = ensure_positive_int(sample_rate, "sample_rate")
    low_hz = ensure_non_negative_float(low_hz, "low_hz")
    high_hz = ensure_non_negative_float(high_hz, "high_hz")
    if high_hz <= low_hz:
        raise ValueError("high_hz must be greater than low_hz")

    s = to_1d_float_array(signal)
    if len(s) == 0:
        return []
    spec = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(len(s), d=1.0 / sample_rate)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    filtered = np.zeros_like(spec)
    filtered[mask] = spec[mask]
    return np.fft.irfft(filtered, n=len(s)).astype(np.float64).tolist()


__all__ = ["moving_average", "median_filter", "remove_dc", "normalize_peak", "fft_bandpass"]
