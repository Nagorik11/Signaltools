"""Complex and analytic multichannel signal helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class ComplexChannelResult:
    real: list[list[float]]
    imag: list[list[float]]
    magnitude: list[list[float]]
    phase: list[list[float]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _as_channel_matrix(features: list[list[float]] | list[float] | np.ndarray, complex_ok: bool = False) -> np.ndarray:
    dtype = np.complex128 if complex_ok else np.float64
    x = np.asarray(features, dtype=dtype)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("features must be a 2D array-like of shape (samples, channels)")
    return x



def _analytic_1d(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    X = np.fft.fft(x)
    h = np.zeros(n, dtype=np.float64)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(X * h)



def analytic_signal_multichannel(features: list[list[float]] | list[float] | np.ndarray) -> ComplexChannelResult:
    """Compute an analytic signal independently for each channel."""
    x = _as_channel_matrix(features)
    analytic = np.stack([_analytic_1d(x[:, c]) for c in range(x.shape[1])], axis=1)
    return ComplexChannelResult(
        real=np.real(analytic).tolist(),
        imag=np.imag(analytic).tolist(),
        magnitude=np.abs(analytic).tolist(),
        phase=np.angle(analytic).tolist(),
        meta={"channels": int(x.shape[1]), "samples": int(x.shape[0]), "analytic": True},
    )



def complex_channel_mix(
    features: list[list[complex]] | list[complex] | np.ndarray,
    mix_matrix: list[list[complex]] | np.ndarray | None = None,
    bias: list[complex] | np.ndarray | None = None,
    residual: bool = False,
    out_channels: int | None = None,
    mix_strength: float = 0.15,
) -> ComplexChannelResult:
    """Apply dense complex-valued channel mixing."""
    x = _as_channel_matrix(features, complex_ok=True)
    in_channels = x.shape[1]
    if mix_matrix is None:
        out_channels = in_channels if out_channels is None else int(out_channels)
        W = np.zeros((out_channels, in_channels), dtype=np.complex128)
        for i in range(out_channels):
            W[i, i % in_channels] = 1.0 + 0.0j
        W = (1.0 - mix_strength) * W + mix_strength * np.ones((out_channels, in_channels), dtype=np.complex128) / max(in_channels, 1)
    else:
        W = np.asarray(mix_matrix, dtype=np.complex128)
        if W.ndim != 2 or W.shape[1] != in_channels:
            raise ValueError("mix_matrix must have shape (out_channels, in_channels)")
        out_channels = W.shape[0]
    b = np.zeros(out_channels, dtype=np.complex128) if bias is None else np.asarray(bias, dtype=np.complex128)
    if b.shape != (out_channels,):
        raise ValueError("bias must have shape (out_channels,)")
    y = x @ W.T + b
    if residual and y.shape == x.shape:
        y = y + x
    return ComplexChannelResult(
        real=np.real(y).tolist(),
        imag=np.imag(y).tolist(),
        magnitude=np.abs(y).tolist(),
        phase=np.angle(y).tolist(),
        meta={"in_channels": int(in_channels), "out_channels": int(out_channels), "residual": residual, "complex": True},
    )


__all__ = ["ComplexChannelResult", "analytic_signal_multichannel", "complex_channel_mix"]
