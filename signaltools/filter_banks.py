"""Multiband filter-bank helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .filter_design import apply_fir, fir_lowpass
from .multirate import decimate, interpolate
from .utils import ensure_positive_int, to_1d_float_array


@dataclass
class FilterBankResult:
    subbands: list[list[float]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def haar_analysis_bank(signal: list[float] | list[int]) -> FilterBankResult:
    x = to_1d_float_array(signal)
    if x.size == 0:
        return FilterBankResult(subbands=[[], []], meta={"type": "haar_analysis"})
    lp = np.array([1.0, 1.0], dtype=np.float64) / np.sqrt(2.0)
    hp = np.array([1.0, -1.0], dtype=np.float64) / np.sqrt(2.0)
    low = np.convolve(x, lp, mode="same")[::2]
    high = np.convolve(x, hp, mode="same")[::2]
    return FilterBankResult(subbands=[low.tolist(), high.tolist()], meta={"type": "haar_analysis"})


def haar_synthesis_bank(low: list[float] | list[int], high: list[float] | list[int]) -> list[float]:
    l = to_1d_float_array(low, name="low")
    h = to_1d_float_array(high, name="high")
    n = max(len(l), len(h))
    up_l = np.zeros(2 * n, dtype=np.float64)
    up_h = np.zeros(2 * n, dtype=np.float64)
    up_l[::2][: len(l)] = l
    up_h[::2][: len(h)] = h
    lp = np.array([1.0, 1.0], dtype=np.float64) / np.sqrt(2.0)
    hp = np.array([1.0, -1.0], dtype=np.float64) / np.sqrt(2.0)
    y = np.convolve(up_l, lp, mode="same") + np.convolve(up_h, hp, mode="same")
    return y.tolist()


def uniform_filter_bank(signal: list[float] | list[int], bands: int = 4, fir_taps: int = 31) -> FilterBankResult:
    bands = ensure_positive_int(bands, "bands")
    x = to_1d_float_array(signal)
    if x.size == 0:
        return FilterBankResult(subbands=[], meta={"type": "uniform_filter_bank", "bands": bands})
    subbands: list[list[float]] = []
    for k in range(bands):
        low = k / (2.0 * bands)
        high = (k + 1) / (2.0 * bands)
        cutoff_low = max(low, 1e-3)
        cutoff_high = min(high, 0.499)
        if k == 0:
            filt = fir_lowpass(fir_taps, cutoff_high, 1, window="hamming")
        else:
            from .filter_design import fir_bandpass
            filt = fir_bandpass(fir_taps, cutoff_low, cutoff_high, 1, window="hamming")
        band = apply_fir(x.tolist(), filt)
        subbands.append(decimate(band, bands, fir_taps=max(15, fir_taps // 2)))
    return FilterBankResult(subbands=subbands, meta={"type": "uniform_filter_bank", "bands": bands, "fir_taps": fir_taps})


def reconstruct_uniform_filter_bank(subbands: list[list[float]], bands: int = 4, fir_taps: int = 31) -> list[float]:
    bands = ensure_positive_int(bands, "bands")
    if not subbands:
        return []
    reconstructed = None
    for band in subbands:
        up = np.asarray(interpolate(band, bands, fir_taps=max(15, fir_taps // 2)), dtype=np.float64)
        reconstructed = up if reconstructed is None else reconstructed + up
    return reconstructed.tolist() if reconstructed is not None else []


__all__ = ["FilterBankResult", "haar_analysis_bank", "haar_synthesis_bank", "uniform_filter_bank", "reconstruct_uniform_filter_bank"]
