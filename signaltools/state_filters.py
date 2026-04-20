"""State-estimation and statistical filters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .utils import ensure_non_negative_float, ensure_positive_int, to_1d_float_array


@dataclass
class KalmanFilterResult:
    estimates: list[float]
    gains: list[float]
    errors: list[float]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WienerFilterResult:
    filtered: list[float]
    local_mean: list[float]
    local_variance: list[float]
    noise_variance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pad(signal: np.ndarray, pad: int, mode: str) -> np.ndarray:
    return np.pad(signal, (pad, pad), mode=mode)


def wiener_filter_1d(
    signal: list[float] | list[int],
    window_size: int = 5,
    noise_variance: float | None = None,
    pad_mode: str = "reflect",
) -> WienerFilterResult:
    """Apply a simple local-statistics Wiener filter."""
    x = to_1d_float_array(signal)
    window_size = ensure_positive_int(window_size, "window_size")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if x.size == 0:
        return WienerFilterResult(filtered=[], local_mean=[], local_variance=[], noise_variance=0.0)

    half = window_size // 2
    padded = _pad(x, half, pad_mode)
    local_mean = np.zeros_like(x)
    local_var = np.zeros_like(x)

    for i in range(len(x)):
        window = padded[i : i + window_size]
        local_mean[i] = np.mean(window)
        local_var[i] = np.var(window)

    if noise_variance is None:
        noise_variance = float(np.mean(local_var))
    else:
        noise_variance = ensure_non_negative_float(noise_variance, "noise_variance")

    gain = np.maximum(local_var - noise_variance, 0.0) / np.maximum(local_var, 1e-12)
    y = local_mean + gain * (x - local_mean)
    return WienerFilterResult(
        filtered=y.tolist(),
        local_mean=local_mean.tolist(),
        local_variance=local_var.tolist(),
        noise_variance=float(noise_variance),
    )


def kalman_filter_1d(
    signal: list[float] | list[int],
    process_variance: float = 1e-5,
    measurement_variance: float = 1e-2,
    initial_estimate: float | None = None,
    initial_error: float = 1.0,
) -> KalmanFilterResult:
    """Run a scalar Kalman filter over a 1D measurement sequence."""
    z = to_1d_float_array(signal)
    process_variance = ensure_non_negative_float(process_variance, "process_variance")
    measurement_variance = ensure_non_negative_float(measurement_variance, "measurement_variance")
    initial_error = ensure_non_negative_float(initial_error, "initial_error")

    if z.size == 0:
        return KalmanFilterResult(estimates=[], gains=[], errors=[], meta={"process_variance": process_variance, "measurement_variance": measurement_variance})

    estimate = float(z[0] if initial_estimate is None else initial_estimate)
    error = float(initial_error)
    estimates: list[float] = []
    gains: list[float] = []
    errors: list[float] = []

    for measurement in z:
        error = error + process_variance
        gain = error / (error + measurement_variance)
        estimate = estimate + gain * (float(measurement) - estimate)
        error = (1.0 - gain) * error
        estimates.append(estimate)
        gains.append(gain)
        errors.append(error)

    return KalmanFilterResult(
        estimates=estimates,
        gains=gains,
        errors=errors,
        meta={"process_variance": process_variance, "measurement_variance": measurement_variance},
    )


__all__ = ["KalmanFilterResult", "WienerFilterResult", "kalman_filter_1d", "wiener_filter_1d"]
