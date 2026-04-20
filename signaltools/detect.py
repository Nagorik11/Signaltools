"""Event, peak, and anomaly detection utilities."""

from __future__ import annotations

import numpy as np

from .utils import ensure_non_negative_float, ensure_positive_int, to_1d_float_array


def threshold_events(signal: list[float], threshold: float) -> list[dict]:
    """Return contiguous event regions whose absolute amplitude exceeds a threshold."""
    threshold = ensure_non_negative_float(threshold, "threshold")
    s = to_1d_float_array(signal)
    events: list[dict] = []
    active = False
    current_indices: list[int] = []
    for i, value in enumerate(s):
        if not active and abs(value) >= threshold:
            active = True
            current_indices = [i]
        elif active and abs(value) >= threshold:
            current_indices.append(i)
        elif active and abs(value) < threshold:
            if current_indices:
                events.append({"start": current_indices[0], "end": current_indices[-1], "length": len(current_indices)})
            active = False
            current_indices = []
    if active and current_indices:
        events.append({"start": current_indices[0], "end": current_indices[-1], "length": len(current_indices)})
    return events


def adaptive_threshold(signal: list[float], z: float = 2.0) -> float:
    """Estimate an amplitude threshold from the absolute signal distribution."""
    z = ensure_non_negative_float(z, "z")
    s = to_1d_float_array(signal)
    if len(s) == 0:
        return 0.0
    return float(np.mean(np.abs(s)) + z * np.std(np.abs(s)))


def adaptive_events(signal: list[float], z: float = 2.0) -> list[dict]:
    """Detect events using an adaptive z-score derived threshold."""
    return threshold_events(signal, adaptive_threshold(signal, z=z))


def local_peaks(signal: list[float], min_height: float = 0.0, min_distance: int = 1) -> list[dict]:
    """Find local maxima with optional minimum height and spacing."""
    min_height = ensure_non_negative_float(min_height, "min_height")
    min_distance = ensure_positive_int(min_distance, "min_distance")
    s = to_1d_float_array(signal)
    if len(s) < 3:
        return []

    peaks: list[dict] = []
    last_kept = -min_distance
    for i in range(1, len(s) - 1):
        if s[i] > s[i - 1] and s[i] >= s[i + 1] and s[i] >= min_height:
            candidate = {"index": i, "value": float(s[i])}
            if i - last_kept >= min_distance:
                peaks.append(candidate)
                last_kept = i
            elif peaks and candidate["value"] > peaks[-1]["value"]:
                peaks[-1] = candidate
                last_kept = i
    return peaks


def anomaly_score(signal: list[float]) -> list[float]:
    """Return absolute z-score anomaly scores for each sample."""
    s = to_1d_float_array(signal)
    if len(s) == 0:
        return []
    mu = float(np.mean(s))
    std = float(np.std(s))
    if std == 0.0:
        return [0.0 for _ in s]
    return np.abs((s - mu) / std).astype(np.float64).tolist()


def onset_strength(signal: list[float]) -> list[float]:
    """Compute a simple positive-difference onset strength envelope."""
    s = to_1d_float_array(signal)
    if len(s) < 2:
        return []
    diff = np.diff(np.abs(s), prepend=np.abs(s[0]))
    diff[diff < 0] = 0
    return diff.astype(np.float64).tolist()


__all__ = [
    "threshold_events",
    "adaptive_threshold",
    "adaptive_events",
    "local_peaks",
    "anomaly_score",
    "onset_strength",
]
