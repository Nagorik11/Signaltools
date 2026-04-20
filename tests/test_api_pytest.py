from __future__ import annotations

import signaltools as st


def demo_signal() -> list[float]:
    return [0.1, 0.4, -0.2, 0.8, -0.1, 0.2] * 100


def test_analyze_signal_advanced_returns_expected_sections() -> None:
    analysis = st.analyze_signal_advanced(demo_signal(), sample_rate=1000, frame_size=64, hop_size=32)
    payload = analysis.to_dict()
    assert payload["summary"]["samples"] > 0
    assert "spectral" in payload
    assert "fingerprint" in payload


def test_fingerprint_has_matching_labels_and_values() -> None:
    fp = st.fingerprint_engine(demo_signal(), sample_rate=1000)
    assert len(fp.vector) == len(fp.labels)
    assert len(fp.vector) >= 10


def test_compare_fingerprints_on_same_signal_is_high_similarity() -> None:
    fp_a = st.fingerprint_engine(demo_signal(), sample_rate=1000)
    fp_b = st.fingerprint_engine(demo_signal(), sample_rate=1000)
    comparison = st.compare_fingerprints(fp_a, fp_b)
    assert comparison["cosine_similarity"] >= 0.999


def test_cli_level_helpers_are_exported() -> None:
    assert callable(st.configure_logging)
    assert st.get_logger("signaltools.test").name == "signaltools.test"
