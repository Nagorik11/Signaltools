from __future__ import annotations

import math

import numpy as np
import pytest

import signaltools as st
from signaltools.bridge import reconstruct_signal_from_signature
from signaltools.exceptions import SignalValidationError
from signaltools.filters import normalize_peak
from signaltools.modulate import amplitude_modulation, frequency_modulation


@pytest.fixture
def waveform() -> list[float]:
    t = np.linspace(0, 1, 512, endpoint=False)
    return (0.8 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 120 * t)).tolist()


def test_features_and_derivatives(waveform: list[float]) -> None:
    features = st.frame_feature_vector(waveform[:128])
    assert features["rms"] > 0
    assert len(st.first_derivative(waveform)) == len(waveform) - 1
    assert len(st.second_derivative(waveform)) == len(waveform) - 2


def test_framing_and_standardization(waveform: list[float]) -> None:
    cfg = st.FrameConfig(frame_size=64, hop_size=32, pad_end=True, window="hann")
    frames = st.frame_signal(waveform, cfg)
    assert len(frames) > 0
    standardized = st.standardize_signal(waveform)
    assert abs(float(np.mean(standardized))) < 1e-6


def test_spectral_suite(waveform: list[float]) -> None:
    dft = st.dft(waveform, mode="rfft")
    assert len(dft["magnitude"]) > 0
    assert len(st.frequency_axis(len(waveform), 1000)) == len(dft["magnitude"])
    assert st.spectral_energy(dft["magnitude"]) > 0
    assert 0 <= st.spectral_flatness(dft["magnitude"]) <= 1
    assert st.spectral_centroid(waveform, sample_rate=1000) > 0
    assert st.spectral_bandwidth(waveform, sample_rate=1000) >= 0
    assert st.spectral_rolloff(waveform, sample_rate=1000) > 0
    assert st.band_energy(waveform, 1000, 40, 140) > 0
    assert len(st.power_spectrum(waveform)) == len(dft["magnitude"])
    assert len(st.stft(waveform, frame_size=64, hop_size=32)) > 0
    assert len(st.spectrogram_matrix(waveform, frame_size=64, hop_size=32)) > 0
    assert len(st.autocorrelation(waveform)) == len(waveform)
    assert st.estimate_pitch(waveform, sample_rate=1000, min_hz=20, max_hz=200) > 0


def test_detection_suite(waveform: list[float]) -> None:
    threshold = st.adaptive_threshold(waveform)
    assert threshold >= 0
    assert isinstance(st.threshold_events(waveform, 0.2), list)
    assert isinstance(st.adaptive_events(waveform), list)
    assert len(st.local_peaks(waveform, min_height=0.2, min_distance=5)) > 0
    assert len(st.anomaly_score(waveform)) == len(waveform)
    assert len(st.onset_strength(waveform)) == len(waveform)


def test_filters_and_modulation(waveform: list[float]) -> None:
    assert len(st.moving_average(waveform, 5)) == len(waveform)
    assert len(st.median_filter(waveform, 5)) == len(waveform)
    assert len(st.remove_dc(waveform)) == len(waveform)
    assert max(abs(x) for x in normalize_peak(waveform)) == pytest.approx(1.0)
    assert len(st.fft_bandpass(waveform, 1000, 20, 200)) == len(waveform)
    mod = [0.1] * len(waveform)
    assert len(amplitude_modulation(waveform, mod)) == len(waveform)
    assert len(frequency_modulation(100.0, mod, sample_rate=1000, index=0.5)) == len(waveform)


def test_validation_errors() -> None:
    with pytest.raises(SignalValidationError):
        st.frame_signal([1, 2, 3], st.FrameConfig(frame_size=0, hop_size=1))
    with pytest.raises(SignalValidationError):
        st.local_peaks([1, 2, 3], min_distance=0)


def test_bitlayer_and_bridge(waveform: list[float]) -> None:
    payload = np.asarray(waveform, dtype=np.float32).tobytes()
    bit = st.analyze_bitlayer(payload)
    assert "compact" in bit
    signature = st.signal_signature(waveform, frame_size=64, hop_size=32)
    assert len(st.signature_to_glyph_vector(signature)) > 0
    layered = st.analyze_signal_layered(waveform, source_type="numeric")
    assert layered.classification["signal_family"]
    rebuilt = reconstruct_signal_from_signature(signature, duration=0.1, sample_rate=1000)
    assert len(rebuilt) == 100


def test_compare_distance_helpers(waveform: list[float]) -> None:
    fp_a = st.fingerprint_engine(waveform, sample_rate=1000, frame_size=64, hop_size=32)
    fp_b = st.fingerprint_engine(waveform, sample_rate=1000, frame_size=64, hop_size=32)
    assert st.cosine_similarity(fp_a.vector, fp_b.vector) == pytest.approx(1.0)
    assert st.euclidean_distance(fp_a.vector, fp_b.vector) == pytest.approx(0.0)
    assert st.compare_fingerprints(fp_a, fp_b)["euclidean_distance"] == pytest.approx(0.0)
