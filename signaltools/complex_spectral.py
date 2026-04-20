"""Complex spectral operators for multichannel analytic signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class ComplexSpectralResult:
    real: list[list[float]]
    imag: list[list[float]]
    magnitude: list[list[float]]
    phase: list[list[float]]
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



def complex_dft_multichannel(features: list | np.ndarray) -> ComplexSpectralResult:
    """Compute a complex DFT independently for each channel."""
    x = _as_complex_matrix(features)
    spec = np.fft.fft(x, axis=0)
    return ComplexSpectralResult(
        real=np.real(spec).tolist(),
        imag=np.imag(spec).tolist(),
        magnitude=np.abs(spec).tolist(),
        phase=np.angle(spec).tolist(),
        meta={"samples": int(x.shape[0]), "channels": int(x.shape[1]), "domain": "frequency"},
    )



def complex_spectral_mask(features: list | np.ndarray, mask: list | np.ndarray | None = None, phase_shift: float = 0.0) -> ComplexSpectralResult:
    """Apply a complex spectral mask and optional global phase shift."""
    x = _as_complex_matrix(features)
    spec = np.fft.fft(x, axis=0)
    if mask is None:
        m = np.ones_like(spec, dtype=np.complex128)
    else:
        m = np.asarray(mask, dtype=np.complex128)
        if m.ndim == 1:
            m = m[:, None]
        if m.shape != spec.shape:
            raise ValueError("mask must match the complex spectrum shape")
    rotated = np.exp(1j * phase_shift)
    y = np.fft.ifft(spec * m * rotated, axis=0)
    return ComplexSpectralResult(
        real=np.real(y).tolist(),
        imag=np.imag(y).tolist(),
        magnitude=np.abs(y).tolist(),
        phase=np.angle(y).tolist(),
        meta={"samples": int(x.shape[0]), "channels": int(x.shape[1]), "phase_shift": float(phase_shift), "masked": mask is not None},
    )



def complex_spectral_shift(features: list | np.ndarray, bins: int = 1) -> ComplexSpectralResult:
    """Circularly shift the complex spectrum of each channel."""
    x = _as_complex_matrix(features)
    spec = np.fft.fft(x, axis=0)
    shifted = np.roll(spec, shift=int(bins), axis=0)
    y = np.fft.ifft(shifted, axis=0)
    return ComplexSpectralResult(
        real=np.real(y).tolist(),
        imag=np.imag(y).tolist(),
        magnitude=np.abs(y).tolist(),
        phase=np.angle(y).tolist(),
        meta={"samples": int(x.shape[0]), "channels": int(x.shape[1]), "bins": int(bins), "shifted": True},
    )


__all__ = ["ComplexSpectralResult", "complex_dft_multichannel", "complex_spectral_mask", "complex_spectral_shift"]
