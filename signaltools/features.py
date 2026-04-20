from __future__ import annotations

from math import sqrt
import numpy as np


def mean(signal: list[float]) -> float:
    return float(np.mean(signal)) if signal else 0.0


def rms(signal: list[float]) -> float:
    return sqrt(sum(x * x for x in signal) / len(signal)) if signal else 0.0


def variance(signal: list[float]) -> float:
    return float(np.var(signal)) if signal else 0.0


def stddev(signal: list[float]) -> float:
    return float(np.std(signal)) if signal else 0.0


def median(signal: list[float]) -> float:
    return float(np.median(signal)) if signal else 0.0


def median_abs_deviation(signal: list[float]) -> float:
    if not signal:
        return 0.0
    med = float(np.median(signal))
    return float(np.median(np.abs(np.asarray(signal) - med)))


def zero_crossing_rate(signal: list[float]) -> float:
    if len(signal) < 2:
        return 0.0
    zc = 0
    for i in range(len(signal) - 1):
        if (signal[i] >= 0 > signal[i + 1]) or (signal[i] < 0 <= signal[i + 1]):
            zc += 1
    return zc / (len(signal) - 1)


def peak_to_peak(signal: list[float]) -> float:
    return (max(signal) - min(signal)) if signal else 0.0


def crest_factor(signal: list[float]) -> float:
    r = rms(signal)
    return max(abs(x) for x in signal) / r if signal and r else 0.0


def signal_energy(signal: list[float]) -> float:
    return float(sum(x * x for x in signal)) if signal else 0.0


def waveform_length(signal: list[float]) -> float:
    return float(sum(abs(signal[i + 1] - signal[i]) for i in range(len(signal) - 1))) if len(signal) >= 2 else 0.0


def dynamic_range(signal: list[float]) -> float:
    if not signal:
        return 0.0
    abs_signal = [abs(x) for x in signal]
    peak = max(abs_signal)
    floor = max(min(v for v in abs_signal if v > 0), 1e-12) if any(v > 0 for v in abs_signal) else 1e-12
    return float(20.0 * np.log10(peak / floor)) if peak > 0 else 0.0


def skewness(signal: list[float]) -> float:
    if len(signal) < 2:
        return 0.0
    s = np.asarray(signal, dtype=np.float64)
    mu = np.mean(s)
    std = np.std(s)
    if std == 0:
        return 0.0
    return float(np.mean(((s - mu) / std) ** 3))


def kurtosis(signal: list[float]) -> float:
    if len(signal) < 2:
        return 0.0
    s = np.asarray(signal, dtype=np.float64)
    mu = np.mean(s)
    std = np.std(s)
    if std == 0:
        return 0.0
    return float(np.mean(((s - mu) / std) ** 4))


def first_derivative(signal: list[float]) -> list[float]:
    return [signal[i + 1] - signal[i] for i in range(len(signal) - 1)] if len(signal) >= 2 else []


def second_derivative(signal: list[float]) -> list[float]:
    return first_derivative(first_derivative(signal))


def frame_feature_vector(frame: list[float]) -> dict:
    return {
        "mean": round(mean(frame), 6),
        "median": round(median(frame), 6),
        "variance": round(variance(frame), 6),
        "stddev": round(stddev(frame), 6),
        "mad": round(median_abs_deviation(frame), 6),
        "rms": round(rms(frame), 6),
        "energy": round(signal_energy(frame), 6),
        "zcr": round(zero_crossing_rate(frame), 6),
        "peak_to_peak": round(peak_to_peak(frame), 6),
        "crest_factor": round(crest_factor(frame), 6),
        "waveform_length": round(waveform_length(frame), 6),
        "dynamic_range_db": round(dynamic_range(frame), 6),
        "skewness": round(skewness(frame), 6),
        "kurtosis": round(kurtosis(frame), 6),
    }


__all__ = [
    "mean",
    "rms",
    "variance",
    "stddev",
    "median",
    "median_abs_deviation",
    "zero_crossing_rate",
    "peak_to_peak",
    "crest_factor",
    "signal_energy",
    "waveform_length",
    "dynamic_range",
    "skewness",
    "kurtosis",
    "first_derivative",
    "second_derivative",
    "frame_feature_vector",
]
