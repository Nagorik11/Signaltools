"""Learnable-style complex operators in the time-frequency domain."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .utils import ensure_positive_int


@dataclass
class ComplexLearnableTFResult:
    time_real: list[list[float]]
    time_imag: list[list[float]]
    stft_in_real: list
    stft_in_imag: list
    stft_out_real: list
    stft_out_imag: list
    magnitude_out: list
    phase_out: list
    params: dict[str, Any]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _as_complex_matrix(features: list | np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.complex128)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("features must be a 2D array-like of shape (samples, channels)")
    return x



def _frame_count(n: int, frame_size: int, hop_size: int) -> int:
    if n <= frame_size:
        return 1
    return 1 + (n - frame_size) // hop_size



def _window(frame_size: int, window: str) -> np.ndarray:
    if window == "hann":
        return np.hanning(frame_size).astype(np.float64)
    if window == "rect":
        return np.ones(frame_size, dtype=np.float64)
    raise ValueError("Unsupported window")



def _broadcast_param(param: list | np.ndarray | None, frame_size: int, channels: int, default: complex) -> np.ndarray:
    if param is None:
        return np.full((frame_size, channels), default, dtype=np.complex128)
    x = np.asarray(param, dtype=np.complex128)
    if x.ndim == 0:
        return np.full((frame_size, channels), complex(x), dtype=np.complex128)
    if x.ndim == 1:
        if len(x) == frame_size:
            return np.repeat(x[:, None], channels, axis=1)
        if len(x) == channels:
            return np.repeat(x[None, :], frame_size, axis=0)
    if x.shape == (frame_size, channels):
        return x.astype(np.complex128)
    raise ValueError("parameter must be scalar, (frame_size,), (channels,) or (frame_size, channels)")



def _apply_complex_activation(x: np.ndarray, activation: str) -> np.ndarray:
    if activation == "linear":
        return x
    mag = np.abs(x)
    phase = np.exp(1j * np.angle(x))
    if activation == "tanh":
        return np.tanh(mag) * phase
    if activation == "sigmoid_mag":
        return (1.0 / (1.0 + np.exp(-mag))) * phase
    if activation == "relu_mag":
        return np.maximum(mag, 0.0) * phase
    raise ValueError("Unsupported activation")



def complex_learnable_tf_operator(
    features: list | np.ndarray,
    frame_size: int = 256,
    hop_size: int = 128,
    gain: list | np.ndarray | None = None,
    bias: list | np.ndarray | None = None,
    phase_shift: float = 0.0,
    activation: str = "linear",
    residual: bool = True,
    window: str = "hann",
) -> ComplexLearnableTFResult:
    """Apply a learnable-style complex affine transform in the STFT domain."""
    x = _as_complex_matrix(features)
    frame_size = ensure_positive_int(frame_size, "frame_size")
    hop_size = ensure_positive_int(hop_size, "hop_size")
    w = _window(frame_size, window)
    n_frames = _frame_count(x.shape[0], frame_size, hop_size)
    gain_arr = _broadcast_param(gain, frame_size, x.shape[1], 1.0 + 0.0j)
    bias_arr = _broadcast_param(bias, frame_size, x.shape[1], 0.0 + 0.0j)
    stft_in = np.zeros((n_frames, frame_size, x.shape[1]), dtype=np.complex128)
    stft_out = np.zeros_like(stft_in)
    recon = np.zeros_like(x, dtype=np.complex128)
    norm = np.zeros(x.shape[0], dtype=np.float64)
    rot = np.exp(1j * phase_shift)
    for frame_idx in range(n_frames):
        start = frame_idx * hop_size
        end = min(start + frame_size, x.shape[0])
        frame = np.zeros((frame_size, x.shape[1]), dtype=np.complex128)
        frame[: end - start] = x[start:end]
        frame = frame * w[:, None]
        spec = np.fft.fft(frame, axis=0)
        stft_in[frame_idx] = spec
        transformed = spec * gain_arr + bias_arr
        transformed = transformed * rot
        transformed = _apply_complex_activation(transformed, activation)
        if residual:
            transformed = transformed + spec
        stft_out[frame_idx] = transformed
        time_frame = np.fft.ifft(transformed, axis=0)
        valid = end - start
        recon[start:end] += time_frame[:valid]
        norm[start:end] += w[:valid] ** 2
    recon = recon / np.maximum(norm[:, None], 1e-12)
    return ComplexLearnableTFResult(
        time_real=np.real(recon).tolist(),
        time_imag=np.imag(recon).tolist(),
        stft_in_real=np.real(stft_in).tolist(),
        stft_in_imag=np.imag(stft_in).tolist(),
        stft_out_real=np.real(stft_out).tolist(),
        stft_out_imag=np.imag(stft_out).tolist(),
        magnitude_out=np.abs(stft_out).tolist(),
        phase_out=np.angle(stft_out).tolist(),
        params={
            "gain_real": np.real(gain_arr).tolist(),
            "gain_imag": np.imag(gain_arr).tolist(),
            "bias_real": np.real(bias_arr).tolist(),
            "bias_imag": np.imag(bias_arr).tolist(),
            "phase_shift": float(phase_shift),
            "activation": activation,
            "residual": residual,
        },
        meta={
            "samples": int(x.shape[0]),
            "channels": int(x.shape[1]),
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "frames": int(n_frames),
            "window": window,
        },
    )



def complex_learnable_tf_stack(
    features: list | np.ndarray,
    depth: int = 2,
    frame_size: int = 256,
    hop_size: int = 128,
    gain: list | np.ndarray | None = None,
    bias: list | np.ndarray | None = None,
    phase_step: float = 0.0,
    activation: str = "linear",
    residual: bool = True,
    window: str = "hann",
) -> ComplexLearnableTFResult:
    """Apply several learnable-style TF layers in sequence."""
    depth = ensure_positive_int(depth, "depth")
    current = _as_complex_matrix(features)
    last: ComplexLearnableTFResult | None = None
    for idx in range(depth):
        last = complex_learnable_tf_operator(
            current,
            frame_size=frame_size,
            hop_size=hop_size,
            gain=gain,
            bias=bias,
            phase_shift=phase_step * idx,
            activation=activation,
            residual=residual,
            window=window,
        )
        current = np.asarray(last.time_real, dtype=np.float64) + 1j * np.asarray(last.time_imag, dtype=np.float64)
    assert last is not None
    last.meta["depth"] = depth
    return last


__all__ = [
    "ComplexLearnableTFResult",
    "complex_learnable_tf_operator",
    "complex_learnable_tf_stack",
]
