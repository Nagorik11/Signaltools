"""Complex STFT and frame-domain operators for multichannel signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .utils import ensure_positive_int


@dataclass
class ComplexFrameResult:
    time_real: list[list[float]]
    time_imag: list[list[float]]
    stft_real: list
    stft_imag: list
    magnitude: list
    phase: list
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



def complex_stft_multichannel(
    features: list | np.ndarray,
    frame_size: int = 256,
    hop_size: int = 128,
    window: str = "hann",
) -> ComplexFrameResult:
    """Compute a complex STFT for each channel independently."""
    x = _as_complex_matrix(features)
    frame_size = ensure_positive_int(frame_size, "frame_size")
    hop_size = ensure_positive_int(hop_size, "hop_size")
    if window == "hann":
        w = np.hanning(frame_size).astype(np.float64)
    else:
        w = np.ones(frame_size, dtype=np.float64)
    n_frames = _frame_count(x.shape[0], frame_size, hop_size)
    stft = np.zeros((n_frames, frame_size, x.shape[1]), dtype=np.complex128)
    for frame_idx in range(n_frames):
        start = frame_idx * hop_size
        end = min(start + frame_size, x.shape[0])
        frame = np.zeros((frame_size, x.shape[1]), dtype=np.complex128)
        frame[: end - start] = x[start:end]
        frame = frame * w[:, None]
        stft[frame_idx] = np.fft.fft(frame, axis=0)
    return ComplexFrameResult(
        time_real=np.real(x).tolist(),
        time_imag=np.imag(x).tolist(),
        stft_real=np.real(stft).tolist(),
        stft_imag=np.imag(stft).tolist(),
        magnitude=np.abs(stft).tolist(),
        phase=np.angle(stft).tolist(),
        meta={"samples": int(x.shape[0]), "channels": int(x.shape[1]), "frame_size": int(frame_size), "hop_size": int(hop_size), "frames": int(n_frames), "window": window},
    )



def complex_frame_operator(
    features: list | np.ndarray,
    frame_size: int = 256,
    hop_size: int = 128,
    phase_shift: float = 0.0,
    spectral_mask: list | np.ndarray | None = None,
    window: str = "hann",
) -> ComplexFrameResult:
    """Apply a per-frame spectral operator and reconstruct with overlap-add."""
    x = _as_complex_matrix(features)
    frame_size = ensure_positive_int(frame_size, "frame_size")
    hop_size = ensure_positive_int(hop_size, "hop_size")
    if window == "hann":
        w = np.hanning(frame_size).astype(np.float64)
    else:
        w = np.ones(frame_size, dtype=np.float64)
    n_frames = _frame_count(x.shape[0], frame_size, hop_size)
    stft = np.zeros((n_frames, frame_size, x.shape[1]), dtype=np.complex128)
    recon = np.zeros_like(x, dtype=np.complex128)
    norm = np.zeros(x.shape[0], dtype=np.float64)
    for frame_idx in range(n_frames):
        start = frame_idx * hop_size
        end = min(start + frame_size, x.shape[0])
        frame = np.zeros((frame_size, x.shape[1]), dtype=np.complex128)
        frame[: end - start] = x[start:end]
        frame = frame * w[:, None]
        spec = np.fft.fft(frame, axis=0)
        if spectral_mask is not None:
            m = np.asarray(spectral_mask, dtype=np.complex128)
            if m.ndim == 1:
                m = m[:, None]
            if m.shape != spec.shape:
                raise ValueError("spectral_mask must match each frame spectrum shape")
            spec = spec * m
        spec = spec * np.exp(1j * phase_shift)
        stft[frame_idx] = spec
        time_frame = np.fft.ifft(spec, axis=0)
        valid = end - start
        recon[start:end] += time_frame[:valid]
        norm[start:end] += w[:valid] ** 2
    recon = recon / np.maximum(norm[:, None], 1e-12)
    return ComplexFrameResult(
        time_real=np.real(recon).tolist(),
        time_imag=np.imag(recon).tolist(),
        stft_real=np.real(stft).tolist(),
        stft_imag=np.imag(stft).tolist(),
        magnitude=np.abs(stft).tolist(),
        phase=np.angle(stft).tolist(),
        meta={"samples": int(x.shape[0]), "channels": int(x.shape[1]), "frame_size": int(frame_size), "hop_size": int(hop_size), "frames": int(n_frames), "phase_shift": float(phase_shift), "masked": spectral_mask is not None, "window": window},
    )


__all__ = ["ComplexFrameResult", "complex_stft_multichannel", "complex_frame_operator"]
