from __future__ import annotations

from typing import Any
import numpy as np

from .signal import Signal


class SignalAnalyzer:
    def __init__(self, signal: Signal, window_size: int = 44100):
        self.signal = signal
        self.window_size = window_size

    def get_timeline_analysis(self) -> list[dict[str, Any]]:
        report: list[dict[str, Any]] = []
        data = self.signal.data
        for i in range(0, len(data), self.window_size):
            chunk = data[i : i + self.window_size]
            if len(chunk) < self.window_size:
                break
            sub_sig = Signal(chunk).normalize()
            bit_layer = sub_sig.get_bit_layer()
            report.append(
                {
                    "start_sample": i,
                    "energy": sub_sig.energy,
                    "flatness": sub_sig.flatness,
                    "complexity": bit_layer["bit_entropy"],
                    "dominant_bin": sub_sig.dominant_bins[0]["bin"] if sub_sig.dominant_bins else 0,
                }
            )
        return report

    def generate_summary(self) -> dict[str, Any]:
        timeline = self.get_timeline_analysis()
        if not timeline:
            return {"error": "Signal too short to analyze"}
        energies = [t["energy"] for t in timeline]
        complexities = [t["complexity"] for t in timeline]
        return {
            "total_windows": len(timeline),
            "avg_energy": float(np.mean(energies)),
            "max_energy": float(np.max(energies)),
            "global_complexity": float(np.mean(complexities)),
            "energy_variance": float(np.var(energies)),
        }
