"""Multirate and polyphase utilities."""

from __future__ import annotations

import numpy as np

from .filter_design import apply_fir, fir_lowpass
from .utils import ensure_positive_int, to_1d_float_array


def polyphase_decompose(coeffs: list[float] | list[int], phases: int) -> list[list[float]]:
    phases = ensure_positive_int(phases, "phases")
    h = to_1d_float_array(coeffs, name="coeffs")
    return [h[p::phases].astype(np.float64).tolist() for p in range(phases)]


def decimate(signal: list[float] | list[int], factor: int, fir_taps: int = 31) -> list[float]:
    factor = ensure_positive_int(factor, "factor")
    x = to_1d_float_array(signal)
    if x.size == 0:
        return []
    anti_alias = fir_lowpass(fir_taps, cutoff_hz=0.45 / factor, sample_rate=1, window="hamming")
    filtered = np.asarray(apply_fir(x.tolist(), anti_alias), dtype=np.float64)
    return filtered[::factor].tolist()


def interpolate(signal: list[float] | list[int], factor: int, fir_taps: int = 31) -> list[float]:
    factor = ensure_positive_int(factor, "factor")
    x = to_1d_float_array(signal)
    if x.size == 0:
        return []
    up = np.zeros(len(x) * factor, dtype=np.float64)
    up[::factor] = x
    recon = fir_lowpass(fir_taps, cutoff_hz=0.45 / factor, sample_rate=1, window="hamming")
    y = np.asarray(apply_fir(up.tolist(), recon), dtype=np.float64) * factor
    return y.tolist()


def two_band_analysis_bank(signal: list[float] | list[int], fir_taps: int = 31) -> tuple[list[float], list[float]]:
    x = to_1d_float_array(signal)
    if x.size == 0:
        return [], []
    lp = fir_lowpass(fir_taps, cutoff_hz=0.25, sample_rate=1, window="hamming")
    hp_coeffs = np.asarray(lp.b, dtype=np.float64)
    hp_coeffs = -hp_coeffs
    hp_coeffs[(len(hp_coeffs) - 1) // 2] += 1.0
    low = np.asarray(apply_fir(x.tolist(), lp), dtype=np.float64)[::2]
    high = np.asarray(apply_fir(x.tolist(), hp_coeffs.tolist()), dtype=np.float64)[::2]
    return low.tolist(), high.tolist()


__all__ = ["polyphase_decompose", "decimate", "interpolate", "two_band_analysis_bank"]
