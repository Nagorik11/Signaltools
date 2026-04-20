"""Filter design and application utilities.

This module groups practical digital filters by broad families:
- FIR (windowed-sinc, fractional delay, differentiators)
- IIR (single-pole and biquad sections)
- spectral/specialized helpers (Savitzky-Golay, Hilbert, envelope)
- adaptive (basic LMS)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal

import numpy as np

from .utils import ensure_non_negative_float, ensure_positive_int, to_1d_float_array

WindowName = Literal["rect", "hann", "hamming", "blackman"]


@dataclass
class FIRCoefficients:
    b: list[float]
    kind: str
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IIRCoefficients:
    b: list[float]
    a: list[float]
    kind: str
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveFilterResult:
    output: list[float]
    error: list[float]
    weights: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _window_values(size: int, window: WindowName) -> np.ndarray:
    if size <= 0:
        return np.array([], dtype=np.float64)
    if window == "rect":
        return np.ones(size, dtype=np.float64)
    if window == "hann":
        return np.hanning(size)
    if window == "hamming":
        return np.hamming(size)
    if window == "blackman":
        return np.blackman(size)
    raise ValueError(f"Unsupported window: {window}")


def _normalize_fir(b: np.ndarray) -> np.ndarray:
    total = float(np.sum(b))
    if abs(total) > 1e-12:
        return b / total
    return b


def _sinc_lowpass(num_taps: int, cutoff_hz: float, sample_rate: int, window: WindowName) -> np.ndarray:
    num_taps = ensure_positive_int(num_taps, "num_taps")
    sample_rate = ensure_positive_int(sample_rate, "sample_rate")
    cutoff_hz = ensure_non_negative_float(cutoff_hz, "cutoff_hz")
    if cutoff_hz <= 0 or cutoff_hz >= sample_rate / 2:
        raise ValueError("cutoff_hz must be between 0 and Nyquist")
    n = np.arange(num_taps, dtype=np.float64)
    m = (num_taps - 1) / 2.0
    fc = cutoff_hz / sample_rate
    h = 2.0 * fc * np.sinc(2.0 * fc * (n - m))
    return _normalize_fir(h * _window_values(num_taps, window))


def apply_fir(signal: list[float] | list[int], coeffs: FIRCoefficients | list[float]) -> list[float]:
    x = to_1d_float_array(signal)
    b = np.asarray(coeffs.b if isinstance(coeffs, FIRCoefficients) else coeffs, dtype=np.float64)
    if x.size == 0:
        return []
    return np.convolve(x, b, mode="same").astype(np.float64).tolist()


def apply_iir(signal: list[float] | list[int], coeffs: IIRCoefficients) -> list[float]:
    x = to_1d_float_array(signal)
    if x.size == 0:
        return []
    b = np.asarray(coeffs.b, dtype=np.float64)
    a = np.asarray(coeffs.a, dtype=np.float64)
    if a.size == 0 or abs(a[0]) < 1e-12:
        raise ValueError("Invalid IIR denominator coefficients")

    y = np.zeros_like(x)
    for n in range(len(x)):
        acc = 0.0
        for k in range(len(b)):
            if n - k >= 0:
                acc += b[k] * x[n - k]
        for k in range(1, len(a)):
            if n - k >= 0:
                acc -= a[k] * y[n - k]
        y[n] = acc / a[0]
    return y.astype(np.float64).tolist()


def fir_lowpass(num_taps: int, cutoff_hz: float, sample_rate: int, window: WindowName = "hamming") -> FIRCoefficients:
    b = _sinc_lowpass(num_taps, cutoff_hz, sample_rate, window)
    return FIRCoefficients(b=b.tolist(), kind="fir_lowpass", meta={"cutoff_hz": cutoff_hz, "sample_rate": sample_rate, "window": window})


def fir_highpass(num_taps: int, cutoff_hz: float, sample_rate: int, window: WindowName = "hamming") -> FIRCoefficients:
    lp = _sinc_lowpass(num_taps, cutoff_hz, sample_rate, window)
    hp = -lp
    hp[(num_taps - 1) // 2] += 1.0
    return FIRCoefficients(b=hp.tolist(), kind="fir_highpass", meta={"cutoff_hz": cutoff_hz, "sample_rate": sample_rate, "window": window})


def fir_bandpass(num_taps: int, low_hz: float, high_hz: float, sample_rate: int, window: WindowName = "hamming") -> FIRCoefficients:
    low_hz = ensure_non_negative_float(low_hz, "low_hz")
    high_hz = ensure_non_negative_float(high_hz, "high_hz")
    if high_hz <= low_hz:
        raise ValueError("high_hz must be greater than low_hz")
    lp_high = _sinc_lowpass(num_taps, high_hz, sample_rate, window)
    lp_low = _sinc_lowpass(num_taps, low_hz, sample_rate, window)
    bp = lp_high - lp_low
    return FIRCoefficients(b=bp.tolist(), kind="fir_bandpass", meta={"low_hz": low_hz, "high_hz": high_hz, "sample_rate": sample_rate, "window": window})


def fir_bandstop(num_taps: int, low_hz: float, high_hz: float, sample_rate: int, window: WindowName = "hamming") -> FIRCoefficients:
    bp = np.asarray(fir_bandpass(num_taps, low_hz, high_hz, sample_rate, window).b, dtype=np.float64)
    bs = -bp
    bs[(num_taps - 1) // 2] += 1.0
    return FIRCoefficients(b=bs.tolist(), kind="fir_bandstop", meta={"low_hz": low_hz, "high_hz": high_hz, "sample_rate": sample_rate, "window": window})


def fractional_delay_fir(num_taps: int, delay_samples: float, window: WindowName = "hamming") -> FIRCoefficients:
    num_taps = ensure_positive_int(num_taps, "num_taps")
    delay_samples = float(delay_samples)
    n = np.arange(num_taps, dtype=np.float64)
    m = (num_taps - 1) / 2.0
    h = np.sinc(n - m - delay_samples)
    h *= _window_values(num_taps, window)
    h = _normalize_fir(h)
    return FIRCoefficients(b=h.tolist(), kind="fractional_delay_fir", meta={"delay_samples": delay_samples, "window": window})


def differentiator_fir() -> FIRCoefficients:
    b = np.array([-0.5, 0.0, 0.5], dtype=np.float64)
    return FIRCoefficients(b=b.tolist(), kind="differentiator_fir", meta={"order": 1})


def iir_integrator_leaky(alpha: float = 0.99) -> IIRCoefficients:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return IIRCoefficients(b=[1.0], a=[1.0, -alpha], kind="iir_integrator_leaky", meta={"alpha": alpha})


def _single_pole_alpha(cutoff_hz: float, sample_rate: int) -> float:
    cutoff_hz = ensure_non_negative_float(cutoff_hz, "cutoff_hz")
    sample_rate = ensure_positive_int(sample_rate, "sample_rate")
    if cutoff_hz <= 0 or cutoff_hz >= sample_rate / 2:
        raise ValueError("cutoff_hz must be between 0 and Nyquist")
    dt = 1.0 / sample_rate
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    return dt / (rc + dt)


def iir_lowpass_single_pole(cutoff_hz: float, sample_rate: int) -> IIRCoefficients:
    alpha = _single_pole_alpha(cutoff_hz, sample_rate)
    return IIRCoefficients(b=[alpha], a=[1.0, -(1.0 - alpha)], kind="iir_lowpass_single_pole", meta={"cutoff_hz": cutoff_hz, "sample_rate": sample_rate})


def iir_highpass_single_pole(cutoff_hz: float, sample_rate: int) -> IIRCoefficients:
    alpha = _single_pole_alpha(cutoff_hz, sample_rate)
    hp_alpha = 1.0 - alpha
    return IIRCoefficients(b=[hp_alpha, -hp_alpha], a=[1.0, -hp_alpha], kind="iir_highpass_single_pole", meta={"cutoff_hz": cutoff_hz, "sample_rate": sample_rate})


def _rbj_common(omega0: float, q: float) -> tuple[float, float, float]:
    alpha = math.sin(omega0) / (2.0 * q)
    cos_w0 = math.cos(omega0)
    return alpha, cos_w0, 1.0 + alpha


def _normalize_biquad(b0: float, b1: float, b2: float, a0: float, a1: float, a2: float, kind: str, meta: dict[str, Any]) -> IIRCoefficients:
    return IIRCoefficients(
        b=[b0 / a0, b1 / a0, b2 / a0],
        a=[1.0, a1 / a0, a2 / a0],
        kind=kind,
        meta=meta,
    )


def biquad_lowpass(cutoff_hz: float, sample_rate: int, q: float = 1 / math.sqrt(2)) -> IIRCoefficients:
    omega0 = 2.0 * math.pi * cutoff_hz / sample_rate
    alpha, cos_w0, a0 = _rbj_common(omega0, q)
    b0 = (1 - cos_w0) / 2.0
    b1 = 1 - cos_w0
    b2 = (1 - cos_w0) / 2.0
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return _normalize_biquad(b0, b1, b2, a0, a1, a2, "biquad_lowpass", {"cutoff_hz": cutoff_hz, "sample_rate": sample_rate, "q": q})


def biquad_highpass(cutoff_hz: float, sample_rate: int, q: float = 1 / math.sqrt(2)) -> IIRCoefficients:
    omega0 = 2.0 * math.pi * cutoff_hz / sample_rate
    alpha, cos_w0, a0 = _rbj_common(omega0, q)
    b0 = (1 + cos_w0) / 2.0
    b1 = -(1 + cos_w0)
    b2 = (1 + cos_w0) / 2.0
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return _normalize_biquad(b0, b1, b2, a0, a1, a2, "biquad_highpass", {"cutoff_hz": cutoff_hz, "sample_rate": sample_rate, "q": q})


def biquad_bandpass(center_hz: float, sample_rate: int, q: float = 1.0) -> IIRCoefficients:
    omega0 = 2.0 * math.pi * center_hz / sample_rate
    alpha, cos_w0, a0 = _rbj_common(omega0, q)
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return _normalize_biquad(b0, b1, b2, a0, a1, a2, "biquad_bandpass", {"center_hz": center_hz, "sample_rate": sample_rate, "q": q})


def biquad_notch(center_hz: float, sample_rate: int, q: float = 30.0) -> IIRCoefficients:
    omega0 = 2.0 * math.pi * center_hz / sample_rate
    alpha, cos_w0, a0 = _rbj_common(omega0, q)
    b0 = 1.0
    b1 = -2 * cos_w0
    b2 = 1.0
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return _normalize_biquad(b0, b1, b2, a0, a1, a2, "biquad_notch", {"center_hz": center_hz, "sample_rate": sample_rate, "q": q})


def biquad_allpass(center_hz: float, sample_rate: int, q: float = 0.707) -> IIRCoefficients:
    omega0 = 2.0 * math.pi * center_hz / sample_rate
    alpha, cos_w0, a0 = _rbj_common(omega0, q)
    b0 = 1 - alpha
    b1 = -2 * cos_w0
    b2 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return _normalize_biquad(b0, b1, b2, a0, a1, a2, "biquad_allpass", {"center_hz": center_hz, "sample_rate": sample_rate, "q": q})


def comb_filter_feedforward(delay_samples: int, gain: float = 0.5) -> IIRCoefficients:
    delay_samples = ensure_positive_int(delay_samples, "delay_samples")
    b = np.zeros(delay_samples + 1, dtype=np.float64)
    b[0] = 1.0
    b[-1] = float(gain)
    return IIRCoefficients(b=b.tolist(), a=[1.0], kind="comb_feedforward", meta={"delay_samples": delay_samples, "gain": gain})


def comb_filter_feedback(delay_samples: int, gain: float = 0.5) -> IIRCoefficients:
    delay_samples = ensure_positive_int(delay_samples, "delay_samples")
    a = np.zeros(delay_samples + 1, dtype=np.float64)
    a[0] = 1.0
    a[-1] = -float(gain)
    return IIRCoefficients(b=[1.0], a=a.tolist(), kind="comb_feedback", meta={"delay_samples": delay_samples, "gain": gain})


def savitzky_golay_coefficients(window_length: int, polyorder: int, deriv: int = 0) -> FIRCoefficients:
    window_length = ensure_positive_int(window_length, "window_length")
    polyorder = ensure_positive_int(polyorder, "polyorder")
    deriv = int(deriv)
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")
    if polyorder >= window_length:
        raise ValueError("polyorder must be smaller than window_length")
    if deriv < 0 or deriv > polyorder:
        raise ValueError("deriv must satisfy 0 <= deriv <= polyorder")
    half = window_length // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(x, N=polyorder + 1, increasing=True)
    ATA_inv = np.linalg.pinv(A.T @ A)
    coeffs = ATA_inv @ A.T
    row = coeffs[deriv] * math.factorial(deriv)
    return FIRCoefficients(b=row.tolist(), kind="savitzky_golay", meta={"window_length": window_length, "polyorder": polyorder, "deriv": deriv})


def savitzky_golay_filter(signal: list[float] | list[int], window_length: int, polyorder: int, deriv: int = 0) -> list[float]:
    coeffs = savitzky_golay_coefficients(window_length, polyorder, deriv)
    return apply_fir(signal, coeffs)


def hilbert_transform_fft(signal: list[float] | list[int]) -> list[float]:
    x = to_1d_float_array(signal)
    if x.size == 0:
        return []
    X = np.fft.fft(x)
    h = np.zeros(len(x), dtype=np.float64)
    if len(x) % 2 == 0:
        h[0] = 1.0
        h[len(x) // 2] = 1.0
        h[1 : len(x) // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (len(x) + 1) // 2] = 2.0
    analytic = np.fft.ifft(X * h)
    return np.imag(analytic).astype(np.float64).tolist()


def analytic_signal(signal: list[float] | list[int]) -> list[complex]:
    x = to_1d_float_array(signal)
    y = np.asarray(hilbert_transform_fft(signal), dtype=np.float64)
    if x.size == 0:
        return []
    return (x + 1j * y).tolist()


def envelope(signal: list[float] | list[int]) -> list[float]:
    z = np.asarray(analytic_signal(signal), dtype=np.complex128)
    if z.size == 0:
        return []
    return np.abs(z).astype(np.float64).tolist()


def lms_adaptive_filter(desired: list[float] | list[int], reference: list[float] | list[int], num_taps: int = 8, step_size: float = 0.01) -> AdaptiveFilterResult:
    d = to_1d_float_array(desired, name="desired")
    x = to_1d_float_array(reference, name="reference")
    num_taps = ensure_positive_int(num_taps, "num_taps")
    step_size = ensure_non_negative_float(step_size, "step_size")
    n = min(len(d), len(x))
    if n == 0:
        return AdaptiveFilterResult(output=[], error=[], weights=[0.0] * num_taps)
    w = np.zeros(num_taps, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    e = np.zeros(n, dtype=np.float64)
    padded = np.pad(x[:n], (num_taps - 1, 0))
    for i in range(n):
        u = padded[i : i + num_taps][::-1]
        y[i] = np.dot(w, u)
        e[i] = d[i] - y[i]
        w += 2.0 * step_size * e[i] * u
    return AdaptiveFilterResult(output=y.tolist(), error=e.tolist(), weights=w.tolist())


__all__ = [
    "FIRCoefficients",
    "IIRCoefficients",
    "AdaptiveFilterResult",
    "apply_fir",
    "apply_iir",
    "fir_lowpass",
    "fir_highpass",
    "fir_bandpass",
    "fir_bandstop",
    "fractional_delay_fir",
    "differentiator_fir",
    "iir_integrator_leaky",
    "iir_lowpass_single_pole",
    "iir_highpass_single_pole",
    "biquad_lowpass",
    "biquad_highpass",
    "biquad_bandpass",
    "biquad_notch",
    "biquad_allpass",
    "comb_filter_feedforward",
    "comb_filter_feedback",
    "savitzky_golay_coefficients",
    "savitzky_golay_filter",
    "hilbert_transform_fft",
    "analytic_signal",
    "envelope",
    "lms_adaptive_filter",
]
