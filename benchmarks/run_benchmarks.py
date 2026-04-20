from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import signaltools as st


def generate_signal(n: int, sample_rate: int = 4000) -> list[float]:
    t = np.arange(n) / sample_rate
    rng = np.random.default_rng(42)
    signal = 0.7 * np.sin(2 * np.pi * 120 * t)
    signal += 0.2 * np.sin(2 * np.pi * 340 * t)
    signal += 0.05 * rng.normal(size=n)
    return signal.astype(np.float64).tolist()


def time_call(func, *args, repeat: int = 2, **kwargs) -> dict[str, float | int]:
    durations_ms: list[float] = []
    for _ in range(repeat):
        start = perf_counter()
        func(*args, **kwargs)
        durations_ms.append((perf_counter() - start) * 1000.0)
    return {
        "repeat": repeat,
        "mean_ms": round(mean(durations_ms), 4),
        "min_ms": round(min(durations_ms), 4),
        "max_ms": round(max(durations_ms), 4),
    }


def main() -> int:
    scenarios = {
        "advanced_analysis": lambda s: st.analyze_signal_advanced(s, sample_rate=4000, frame_size=128, hop_size=64),
        "fingerprint": lambda s: st.fingerprint_engine(s, sample_rate=4000, frame_size=128, hop_size=64),
        "spectrogram": lambda s: st.spectrogram_matrix(s, frame_size=128, hop_size=64),
    }

    sizes = {"1k": 1_000, "2k": 2_000, "4k": 4_000}
    results: dict[str, dict] = {
        "environment": {"sample_rate": 4000, "python_entry": "benchmarks/run_benchmarks.py"},
        "benchmarks": {},
    }

    for size_label, n in sizes.items():
        signal = generate_signal(n)
        results["benchmarks"][size_label] = {}
        for name, func in scenarios.items():
            results["benchmarks"][size_label][name] = time_call(func, signal)

    out_path = ROOT / "benchmarks" / "benchmark_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved benchmark results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
