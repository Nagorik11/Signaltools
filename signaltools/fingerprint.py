from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import math
import numpy as np

from .features import frame_feature_vector
from .framing import FrameConfig, detrend_mean, frame_signal, normalize_signal
from .spectral import estimate_pitch, spectral_bandwidth, spectral_centroid, spectral_rolloff
from .detect import adaptive_events, local_peaks


@dataclass
class SignalFingerprint:
    vector: list[float]
    labels: list[str]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DEFAULT_LABELS = [
    "mean",
    "median",
    "variance",
    "stddev",
    "mad",
    "rms",
    "energy",
    "zcr",
    "peak_to_peak",
    "crest_factor",
    "waveform_length",
    "dynamic_range_db",
    "skewness",
    "kurtosis",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff_hz",
    "pitch_hz",
    "event_count",
    "peak_count",
]


def fingerprint_engine(signal: list[float] | list[int], sample_rate: int = 44100, frame_size: int = 256, hop_size: int = 128) -> SignalFingerprint:
    s = detrend_mean(normalize_signal(signal))
    frames = frame_signal(s, FrameConfig(frame_size=frame_size, hop_size=hop_size, pad_end=True, window="hann"))
    if frames:
        features = [frame_feature_vector(f) for f in frames]
        aggregate = {k: float(np.mean([row[k] for row in features])) for k in features[0].keys()}
    else:
        aggregate = {k: 0.0 for k in [
            "mean", "median", "variance", "stddev", "mad", "rms", "energy", "zcr", "peak_to_peak",
            "crest_factor", "waveform_length", "dynamic_range_db", "skewness", "kurtosis"
        ]}

    vector = [
        aggregate["mean"],
        aggregate["median"],
        aggregate["variance"],
        aggregate["stddev"],
        aggregate["mad"],
        aggregate["rms"],
        aggregate["energy"],
        aggregate["zcr"],
        aggregate["peak_to_peak"],
        aggregate["crest_factor"],
        aggregate["waveform_length"],
        aggregate["dynamic_range_db"],
        aggregate["skewness"],
        aggregate["kurtosis"],
        float(spectral_centroid(s, sample_rate=sample_rate)),
        float(spectral_bandwidth(s, sample_rate=sample_rate)),
        float(spectral_rolloff(s, sample_rate=sample_rate)),
        float(estimate_pitch(s, sample_rate=sample_rate)),
        float(len(adaptive_events(s))),
        float(len(local_peaks(s, min_height=0.2, min_distance=max(1, frame_size // 16)))),
    ]

    return SignalFingerprint(
        vector=[round(float(x), 6) for x in vector],
        labels=list(_DEFAULT_LABELS),
        meta={"sample_rate": sample_rate, "frame_size": frame_size, "hop_size": hop_size},
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def euclidean_distance(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(va - vb))


def compare_fingerprints(fp_a: SignalFingerprint, fp_b: SignalFingerprint) -> dict[str, float]:
    return {
        "cosine_similarity": round(cosine_similarity(fp_a.vector, fp_b.vector), 6),
        "euclidean_distance": round(euclidean_distance(fp_a.vector, fp_b.vector), 6),
    }


__all__ = ["SignalFingerprint", "fingerprint_engine", "cosine_similarity", "euclidean_distance", "compare_fingerprints"]
