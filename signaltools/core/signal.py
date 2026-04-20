from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..bitlayer import analyze_bitlayer
from ..features import frame_feature_vector
from ..filters import remove_dc
from ..framing import normalize_signal
from ..spectral import dominant_bins, dft, spectral_energy, spectral_flatness


@dataclass
class Signal:
    data: list[float]

    def __init__(self, data: list[float] | np.ndarray):
        self.data = np.asarray(data, dtype=np.float64).tolist()
        self._spectral_results: dict[str, Any] | None = None
        self.features: dict[str, float] = {}

    def normalize(self) -> "Signal":
        self.data = normalize_signal(remove_dc(self.data))
        self._spectral_results = None
        return self

    def extract_features(self) -> dict[str, float]:
        self.features = frame_feature_vector(self.data)
        return self.features

    def _get_spectral(self) -> dict[str, Any]:
        if self._spectral_results is None:
            self._spectral_results = dft(self.data, mode="rfft")
        return self._spectral_results

    @property
    def dominant_bins(self) -> list[dict]:
        return dominant_bins(self._get_spectral()["magnitude"])

    @property
    def energy(self) -> float:
        return spectral_energy(self._get_spectral()["magnitude"])

    @property
    def flatness(self) -> float:
        return spectral_flatness(self._get_spectral()["magnitude"])

    def get_bit_layer(self) -> dict[str, Any]:
        raw = np.asarray(self.data, dtype=np.float32).tobytes()
        result = analyze_bitlayer(raw)
        signature = result.get("signature", {})
        return {
            "bit_entropy": signature.get("entropy", 0.0),
            "compact": result.get("compact", ""),
        }

    def get_glyph_vector(self) -> list[float]:
        if not self.features:
            self.extract_features()
        return [
            self.features.get("mean", 0.0),
            self.features.get("variance", 0.0),
            self.features.get("rms", 0.0),
            self.energy,
            self.flatness,
            self.get_bit_layer()["bit_entropy"],
        ]
