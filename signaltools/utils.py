"""Shared helpers and input validation utilities."""

from __future__ import annotations

from typing import Iterable
import numpy as np

from .exceptions import SignalValidationError


DEFAULT_EPSILON = 1e-12


def ensure_positive_int(value: int, name: str) -> int:
    """Validate that a parameter is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise SignalValidationError(f"{name} must be a positive integer, got {value!r}")
    return value


def ensure_non_negative_float(value: float, name: str) -> float:
    """Validate that a numeric parameter is non-negative."""
    try:
        value = float(value)
    except (TypeError, ValueError) as e:
        raise SignalValidationError(f"{name} must be numeric, got {value!r}") from e
    if value < 0:
        raise SignalValidationError(f"{name} must be non-negative, got {value!r}")
    return value


def to_1d_float_array(signal: Iterable[float] | np.ndarray | list[int] | list[float], *, name: str = "signal") -> np.ndarray:
    """Convert an input signal to a validated 1D float64 numpy array."""
    try:
        array = np.asarray(list(signal) if not isinstance(signal, np.ndarray) else signal, dtype=np.float64)
    except Exception as e:
        raise SignalValidationError(f"{name} could not be converted to a numeric array") from e

    if array.ndim != 1:
        raise SignalValidationError(f"{name} must be 1-dimensional, got shape {array.shape}")
    if array.size == 0:
        return np.array([], dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise SignalValidationError(f"{name} contains NaN or infinite values")
    return array


def safe_mean(values: list[float]) -> float:
    """Return the mean or 0.0 for empty lists."""
    return float(np.mean(values)) if values else 0.0


def round_float(value: float, digits: int = 6) -> float:
    """Round a numeric value consistently."""
    return round(float(value), digits)
