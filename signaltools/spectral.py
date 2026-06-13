from __future__ import annotations

from math import atan2
from typing import Literal
import numpy as np


SpectrumMode = Literal["fft", "rfft"]


def _safe_array(signal: list[float] | list[int]) -> np.ndarray:
    return np.asarray(signal, dtype=np.float64)


def dft(signal: list[float] | list[int], mode: SpectrumMode = "fft") -> dict:
    s = _safe_array(signal)
    if len(s) == 0:
        return {"real": [], "imag": [], "magnitude": [], "phase": []}
    spectrum = np.fft.rfft(s) if mode == "rfft" else np.fft.fft(s)
    return {
        "real": [round(float(x), 6) for x in spectrum.real],
        "imag": [round(float(x), 6) for x in spectrum.imag],
        "magnitude": [round(float(abs(x)), 6) for x in spectrum],
        "phase": [round(atan2(float(x.imag), float(x.real)) if abs(x) else 0.0, 6) for x in spectrum],
    }


def dominant_bins(magnitude: list[float], top_k: int = 8) -> list[dict]:
    ranked = sorted(({"bin": i, "magnitude": m} for i, m in enumerate(magnitude)), key=lambda x: x["magnitude"], reverse=True)
    return ranked[:top_k]


def spectral_energy(magnitude: list[float]) -> float:
    return round(sum(m * m for m in magnitude), 6)


def spectral_flatness(magnitude: list[float]) -> float:
    vals = [m for m in magnitude if m > 0]
    if not vals:
        return 0.0
    geo = float(np.exp(np.mean(np.log(vals))))
    arith = sum(vals) / len(vals)
    return round(geo / arith, 6) if arith else 0.0


def frequency_axis(n_samples: int, sample_rate: int, real_only: bool = True) -> list[float]:
    if n_samples <= 0:
        return []
    if real_only:
        return np.fft.rfftfreq(n_samples, d=1.0 / sample_rate).tolist()
    return np.fft.fftfreq(n_samples, d=1.0 / sample_rate).tolist()


def power_spectrum(signal: list[float] | list[int]) -> list[float]:
    s = _safe_array(signal)
    if len(s) == 0:
        return []
    spec = np.fft.rfft(s)
    return (np.abs(spec) ** 2).astype(np.float64).tolist()


def spectral_centroid(signal: list[float] | list[int], sample_rate: int = 44100) -> float:
    s = _safe_array(signal)
    if len(s) == 0:
        return 0.0
    mag = np.abs(np.fft.rfft(s))
    freqs = np.fft.rfftfreq(len(s), d=1.0 / sample_rate)
    denom = np.sum(mag)
    return float(np.sum(freqs * mag) / denom) if denom > 0 else 0.0


def spectral_bandwidth(signal: list[float] | list[int], sample_rate: int = 44100) -> float:
    s = _safe_array(signal)
    if len(s) == 0:
        return 0.0
    mag = np.abs(np.fft.rfft(s))
    freqs = np.fft.rfftfreq(len(s), d=1.0 / sample_rate)
    denom = np.sum(mag)
    if denom <= 0:
        return 0.0
    centroid = np.sum(freqs * mag) / denom
    return float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / denom))


def spectral_rolloff(signal: list[float] | list[int], sample_rate: int = 44100, roll_percent: float = 0.85) -> float:
    s = _safe_array(signal)
    if len(s) == 0:
        return 0.0
    power = np.abs(np.fft.rfft(s)) ** 2
    freqs = np.fft.rfftfreq(len(s), d=1.0 / sample_rate)
    total = np.sum(power)
    if total <= 0:
        return 0.0
    threshold = roll_percent * total
    cumulative = np.cumsum(power)
    idx = int(np.searchsorted(cumulative, threshold))
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def band_energy(signal: list[float] | list[int], sample_rate: int, low_hz: float, high_hz: float) -> float:
    s = _safe_array(signal)
    if len(s) == 0 or high_hz <= low_hz:
        return 0.0
    spec = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(len(s), d=1.0 / sample_rate)
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.sum(np.abs(spec[mask]) ** 2))


def stft(signal: list[float] | list[int], frame_size: int = 256, hop_size: int = 128, window: str = "hann") -> list[list[float]]:
    s = _safe_array(signal)
    if len(s) == 0 or frame_size <= 0 or hop_size <= 0:
        return []
    if window == "hann":
        w = np.hanning(frame_size)
    elif window == "hamming":
        w = np.hamming(frame_size)
    elif window == "blackman":
        w = np.blackman(frame_size)
    else:
        w = np.ones(frame_size)

    frames: list[list[float]] = []
    for start in range(0, len(s) - frame_size + 1, hop_size):
        frame = s[start : start + frame_size] * w
        frames.append(np.abs(np.fft.rfft(frame)).astype(np.float64).tolist())
    return frames


def spectrogram_matrix(signal: list[float] | list[int], frame_size: int = 256, hop_size: int = 128, window: str = "hann", log_scale: bool = True) -> list[list[float]]:
    spec = np.asarray(stft(signal, frame_size=frame_size, hop_size=hop_size, window=window), dtype=np.float64)
    if spec.size == 0:
        return []
    if log_scale:
        spec = 20.0 * np.log10(np.maximum(spec, 1e-12))
    return spec.tolist()


def autocorrelation(signal: list[float] | list[int], normalize: bool = True) -> list[float]:
    s = _safe_array(signal)
    if len(s) == 0:
        return []
    s = s - np.mean(s)
    corr = np.correlate(s, s, mode="full")
    corr = corr[len(corr) // 2 :]
    if normalize and corr[0] != 0:
        corr = corr / corr[0]
    return corr.astype(np.float64).tolist()


def estimate_pitch(signal: list[float] | list[int], sample_rate: int = 44100, min_hz: float = 50.0, max_hz: float = 1000.0) -> float:
    ac = np.asarray(autocorrelation(signal, normalize=True), dtype=np.float64)
    if len(ac) < 2:
        return 0.0
    min_lag = max(1, int(sample_rate / max_hz))
    max_lag = min(len(ac) - 1, int(sample_rate / min_hz))
    if max_lag <= min_lag:
        return 0.0
    window = ac[min_lag:max_lag + 1]
    if window.size == 0:  # pragma: no cover
        return 0.0
    lag = min_lag + int(np.argmax(window))
    return float(sample_rate / lag) if lag > 0 else 0.0


def dominant_frequency(signal: list[float] | list[int], sample_rate: int = 44100) -> float:
    s = _safe_array(signal)
    if len(s) == 0:
        return 0.0
    mag = np.abs(np.fft.rfft(s))
    freqs = np.fft.rfftfreq(len(s), d=1.0 / sample_rate)
    idx = int(np.argmax(mag))
    return float(freqs[idx])


def power_spectral_density(
    signal: list[float] | list[int],
    sample_rate: int = 44100,
    window: str = "hann",
    nperseg: int = 256,
) -> tuple[list[float], list[float]]:
    s = _safe_array(signal)
    if len(s) < nperseg:
        fs = np.fft.rfftfreq(len(s), d=1.0 / sample_rate)
        psd = (np.abs(np.fft.rfft(s)) ** 2) / (sample_rate * len(s))
        return fs.tolist(), (2.0 * psd).tolist()
    if window == "hann":
        w = np.hanning(nperseg)
    elif window == "hamming":
        w = np.hamming(nperseg)
    elif window == "blackman":
        w = np.blackman(nperseg)
    else:
        w = np.ones(nperseg)
    scale = 1.0 / (sample_rate * np.sum(w ** 2))
    psd_avg = None
    count = 0
    for start in range(0, len(s) - nperseg + 1, nperseg // 2):
        segment = s[start:start + nperseg] * w
        mag_sq = np.abs(np.fft.rfft(segment)) ** 2
        psd_seg = mag_sq * scale
        if psd_avg is None:
            psd_avg = psd_seg
        else:
            psd_avg += psd_seg
        count += 1
    psd_avg = psd_avg / count if count > 0 else psd_avg
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sample_rate)
    return freqs.tolist(), (2.0 * psd_avg).tolist()


def spectral_entropy(signal: list[float] | list[int]) -> float:
    s = _safe_array(signal)
    if len(s) < 2:
        return 0.0
    mag = np.abs(np.fft.rfft(s))
    psd = mag ** 2
    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    return float(entropy)


def estimate_snr(signal: list[float] | list[int]) -> float:
    s = _safe_array(signal)
    if len(s) < 2:
        return 0.0
    s_detrend = s - np.mean(s)
    power = np.mean(s_detrend ** 2)
    mag = np.abs(np.fft.rfft(s_detrend))
    noise_floor = np.median(mag ** 2)
    snr_db = 10.0 * np.log10(power / noise_floor) if noise_floor > 0 else 0.0
    return round(float(snr_db), 6)


__all__ = [
    "dft",
    "dominant_bins",
    "spectral_energy",
    "spectral_flatness",
    "frequency_axis",
    "power_spectrum",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "band_energy",
    "stft",
    "spectrogram_matrix",
    "autocorrelation",
    "estimate_pitch",
    "dominant_frequency",
    "power_spectral_density",
    "spectral_entropy",
    "estimate_snr",
]
