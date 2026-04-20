"""Multi-head complex time-frequency operators with per-head/per-band parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .complex_learnable_tf import complex_learnable_tf_operator
from .utils import ensure_positive_int


@dataclass
class ComplexMultiHeadTFResult:
    time_real: list[list[float]]
    time_imag: list[list[float]]
    head_outputs_real: list
    head_outputs_imag: list
    band_assignments: list[int]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _make_band_assignments(frame_size: int, num_heads: int) -> np.ndarray:
    bins = np.arange(frame_size)
    return np.floor(num_heads * bins / max(frame_size, 1)).astype(int).clip(0, num_heads - 1)



def _head_param(param: list | np.ndarray | None, head_idx: int, frame_size: int, channels: int, default: complex) -> np.ndarray:
    if param is None:
        return np.full((frame_size, channels), default, dtype=np.complex128)
    x = np.asarray(param, dtype=np.complex128)
    if x.ndim == 0:
        return np.full((frame_size, channels), complex(x), dtype=np.complex128)
    if x.ndim == 1:
        if len(x) > head_idx:
            return np.full((frame_size, channels), complex(x[head_idx]), dtype=np.complex128)
        raise ValueError("per-head vector is shorter than num_heads")
    if x.ndim == 2:
        if x.shape == (frame_size, channels):
            return x
        if x.shape[1] == frame_size and x.shape[0] > head_idx:
            return np.repeat(x[head_idx][:, None], channels, axis=1)
        if x.shape[1] == channels and x.shape[0] > head_idx:
            return np.repeat(x[head_idx][None, :], frame_size, axis=0)
        raise ValueError("2D parameter must be (heads, frame_size), (heads, channels) or (frame_size, channels)")
    if x.ndim == 3 and x.shape[1:] == (frame_size, channels) and x.shape[0] > head_idx:
        return x[head_idx]
    raise ValueError("parameter must be scalar, per-head vector, (heads, frame_size), (heads, channels) or (heads, frame_size, channels)")



def multihead_band_complex_tf_operator(
    features: list | np.ndarray,
    frame_size: int = 256,
    hop_size: int = 128,
    num_heads: int = 4,
    head_gains: list | np.ndarray | None = None,
    head_biases: list | np.ndarray | None = None,
    phase_shifts: list[float] | np.ndarray | None = None,
    activation: str = "linear",
    residual: bool = True,
    combine: str = "mean",
    window: str = "hann",
) -> ComplexMultiHeadTFResult:
    """Apply per-head time-frequency operators, each focused on its own band allocation."""
    x = np.asarray(features, dtype=np.complex128)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("features must be 2D")
    frame_size = ensure_positive_int(frame_size, "frame_size")
    hop_size = ensure_positive_int(hop_size, "hop_size")
    num_heads = ensure_positive_int(num_heads, "num_heads")
    band_assign = _make_band_assignments(frame_size, num_heads)
    phase = np.zeros(num_heads, dtype=float) if phase_shifts is None else np.asarray(phase_shifts, dtype=float)
    if phase.ndim == 0:
        phase = np.full(num_heads, float(phase), dtype=float)
    if len(phase) != num_heads:
        raise ValueError("phase_shifts must have length num_heads")
    head_times = []
    head_specs = []
    for head_idx in range(num_heads):
        gain = _head_param(head_gains, head_idx, frame_size, x.shape[1], 1.0 + 0.0j)
        bias = _head_param(head_biases, head_idx, frame_size, x.shape[1], 0.0 + 0.0j)
        mask = (band_assign == head_idx).astype(np.complex128)[:, None]
        res = complex_learnable_tf_operator(
            x,
            frame_size=frame_size,
            hop_size=hop_size,
            gain=gain * mask,
            bias=bias * mask,
            phase_shift=float(phase[head_idx]),
            activation=activation,
            residual=residual,
            window=window,
        )
        time = np.asarray(res.time_real, dtype=np.float64) + 1j * np.asarray(res.time_imag, dtype=np.float64)
        spec = np.asarray(res.stft_out_real, dtype=np.float64) + 1j * np.asarray(res.stft_out_imag, dtype=np.float64)
        head_times.append(time)
        head_specs.append(spec)
    head_stack = np.stack(head_times, axis=0)
    if combine == "mean":
        out = np.mean(head_stack, axis=0)
    elif combine == "sum":
        out = np.sum(head_stack, axis=0)
    else:
        raise ValueError("combine must be 'mean' or 'sum'")
    return ComplexMultiHeadTFResult(
        time_real=np.real(out).tolist(),
        time_imag=np.imag(out).tolist(),
        head_outputs_real=np.real(np.stack(head_specs, axis=0)).tolist(),
        head_outputs_imag=np.imag(np.stack(head_specs, axis=0)).tolist(),
        band_assignments=band_assign.tolist(),
        meta={
            "samples": int(x.shape[0]),
            "channels": int(x.shape[1]),
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "num_heads": int(num_heads),
            "combine": combine,
            "activation": activation,
            "residual": residual,
            "window": window,
        },
    )



def multihead_band_complex_tf_stack(
    features: list | np.ndarray,
    depth: int = 2,
    **kwargs: Any,
) -> ComplexMultiHeadTFResult:
    """Apply several multi-head banded TF operators in sequence."""
    depth = ensure_positive_int(depth, "depth")
    current = np.asarray(features, dtype=np.complex128)
    if current.ndim == 1:
        current = current[:, None]
    last: ComplexMultiHeadTFResult | None = None
    for _ in range(depth):
        last = multihead_band_complex_tf_operator(current, **kwargs)
        current = np.asarray(last.time_real, dtype=np.float64) + 1j * np.asarray(last.time_imag, dtype=np.float64)
    assert last is not None
    last.meta["depth"] = depth
    return last


__all__ = [
    "ComplexMultiHeadTFResult",
    "multihead_band_complex_tf_operator",
    "multihead_band_complex_tf_stack",
]
