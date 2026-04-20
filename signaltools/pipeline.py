"""High-level orchestration pipeline for advanced signal analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .bitlayer import analyze_bitlayer
from .bridge import analyze_signal_layered, signal_signature
from .detect import adaptive_events, anomaly_score, onset_strength
from .features import frame_feature_vector
from .fingerprint import fingerprint_engine
from .framing import FrameConfig, detrend_mean, frame_signal, normalize_signal
from .logging_utils import get_logger
from .spectral import (
    autocorrelation,
    estimate_pitch,
    spectral_bandwidth,
    spectral_centroid,
    spectral_rolloff,
    spectrogram_matrix,
)

logger = get_logger(__name__)


@dataclass
class AdvancedSignalAnalysis:
    """Structured result returned by `analyze_signal_advanced`."""

    summary: dict[str, Any]
    frames: dict[str, Any]
    time_domain: dict[str, Any]
    spectral: dict[str, Any]
    temporal: dict[str, Any]
    symbolic: dict[str, Any]
    fingerprint: dict[str, Any]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass payload to a plain dictionary."""
        return asdict(self)


def analyze_signal_advanced(
    signal: list[float] | list[int],
    sample_rate: int = 44100,
    frame_size: int = 256,
    hop_size: int = 128,
) -> AdvancedSignalAnalysis:
    """Run the full advanced analysis pipeline over a 1D signal."""
    logger.debug(
        "Starting advanced analysis | sample_rate=%s frame_size=%s hop_size=%s",
        sample_rate,
        frame_size,
        hop_size,
    )
    prepared = detrend_mean(normalize_signal(signal))
    cfg = FrameConfig(frame_size=frame_size, hop_size=hop_size, pad_end=True, window="hann")
    frames = frame_signal(prepared, cfg)
    frame_features = [frame_feature_vector(frame) for frame in frames]

    signature = signal_signature(prepared, frame_size=frame_size, hop_size=hop_size)
    layered = analyze_signal_layered(prepared, source_type="numeric")
    fingerprint = fingerprint_engine(
        prepared,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
    )
    events = adaptive_events(prepared)
    anomalies = anomaly_score(prepared)
    onsets = onset_strength(prepared)
    specgram = spectrogram_matrix(prepared, frame_size=frame_size, hop_size=hop_size)
    ac = autocorrelation(prepared)

    bit_payload = np.asarray(prepared, dtype=np.float32).tobytes()
    bit_analysis = analyze_bitlayer(bit_payload)

    frame_aggregate: dict[str, float] = {}
    if frame_features:
        keys = frame_features[0].keys()
        frame_aggregate = {
            key: round(float(np.mean([row[key] for row in frame_features])), 6)
            for key in keys
        }

    result = AdvancedSignalAnalysis(
        summary={
            "samples": len(prepared),
            "frame_count": len(frames),
            "sample_rate": sample_rate,
        },
        frames={
            "config": {
                "frame_size": frame_size,
                "hop_size": hop_size,
                "window": cfg.window,
                "pad_end": cfg.pad_end,
            },
            "aggregate_features": frame_aggregate,
        },
        time_domain={
            "signature": signature.to_dict(),
            "autocorrelation_preview": [round(float(value), 6) for value in ac[:32]],
        },
        spectral={
            "centroid_hz": round(float(spectral_centroid(prepared, sample_rate=sample_rate)), 6),
            "bandwidth_hz": round(float(spectral_bandwidth(prepared, sample_rate=sample_rate)), 6),
            "rolloff_hz": round(float(spectral_rolloff(prepared, sample_rate=sample_rate)), 6),
            "pitch_hz": round(float(estimate_pitch(prepared, sample_rate=sample_rate)), 6),
            "spectrogram_shape": [len(specgram), len(specgram[0]) if specgram else 0],
            "spectrogram_preview": [
                [round(float(value), 4) for value in row[:8]] for row in specgram[:4]
            ],
        },
        temporal={
            "adaptive_event_count": len(events),
            "events_preview": events[:8],
            "onset_preview": [round(float(value), 6) for value in onsets[:32]],
        },
        symbolic={
            "layered": layered.to_dict(),
            "bitlayer": bit_analysis,
        },
        fingerprint=fingerprint.to_dict(),
        diagnostics={
            "max_anomaly_score": round(float(max(anomalies, default=0.0)), 6),
            "mean_anomaly_score": round(float(np.mean(anomalies)) if anomalies else 0.0, 6),
        },
    )
    logger.debug(
        "Completed advanced analysis | samples=%s frames=%s events=%s",
        len(prepared),
        len(frames),
        len(events),
    )
    return result


__all__ = ["AdvancedSignalAnalysis", "analyze_signal_advanced"]
