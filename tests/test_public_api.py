from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import signaltools as st


class TestSignalToolsPublicAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.signal = [0.1, 0.4, -0.2, 0.8, -0.1, 0.2] * 100

    def test_package_imports(self) -> None:
        self.assertTrue(hasattr(st, "analyze_signal_advanced"))
        self.assertTrue(hasattr(st, "fingerprint_engine"))

    def test_advanced_analysis(self) -> None:
        analysis = st.analyze_signal_advanced(self.signal, sample_rate=1000, frame_size=64, hop_size=32)
        self.assertGreater(analysis.summary["samples"], 0)
        self.assertIn("centroid_hz", analysis.spectral)

    def test_fingerprint(self) -> None:
        fingerprint = st.fingerprint_engine(self.signal, sample_rate=1000)
        self.assertEqual(len(fingerprint.vector), len(fingerprint.labels))

    def test_compare_fingerprints(self) -> None:
        fp_a = st.fingerprint_engine(self.signal, sample_rate=1000)
        fp_b = st.fingerprint_engine(self.signal, sample_rate=1000)
        comparison = st.compare_fingerprints(fp_a, fp_b)
        self.assertGreaterEqual(comparison["cosine_similarity"], 0.999)

    def test_filters(self) -> None:
        filtered = st.moving_average(self.signal, window_size=5)
        self.assertEqual(len(filtered), len(self.signal))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
