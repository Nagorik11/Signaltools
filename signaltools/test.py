"""Demonstration script for the public API."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import signaltools as st


def generate_demo_signal(sample_rate: int = 2000, duration_s: float = 1.5) -> list[float]:
    """Generate a reproducible synthetic signal for demo purposes."""
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    signal = 0.7 * np.sin(2 * np.pi * 120 * t)
    signal += 0.25 * np.sin(2 * np.pi * 280 * t)
    signal += 0.08 * np.random.default_rng(42).normal(size=len(t))
    signal[400:430] += 0.7
    return signal.astype(np.float64).tolist()


def main() -> int:
    """Run the package demo and save a JSON report."""
    print("=== Testing signaltools Library Capabilities ===")
    signal = generate_demo_signal()
    advanced = st.analyze_signal_advanced(signal, sample_rate=2000, frame_size=256, hop_size=128)
    fingerprint = st.fingerprint_engine(signal, sample_rate=2000)

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "advanced_analysis.json"
    out_path.write_text(json.dumps(advanced.to_dict(), indent=2), encoding="utf-8")

    print(f"Samples: {advanced.summary['samples']}")
    print(f"Frames: {advanced.summary['frame_count']}")
    print(f"Spectral centroid: {advanced.spectral['centroid_hz']} Hz")
    print(f"Estimated pitch: {advanced.spectral['pitch_hz']} Hz")
    print(f"Adaptive events: {advanced.temporal['adaptive_event_count']}")
    print(f"Fingerprint length: {len(fingerprint.vector)}")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
