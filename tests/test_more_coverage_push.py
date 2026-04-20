from __future__ import annotations

import struct

import numpy as np
import pytest

import signaltools as st
from signaltools import bitlayer
from signaltools.bridge import analyze_signal_layered
from signaltools.fingerprint import fingerprint_engine
from signaltools.framing import _window_values
from signaltools.io import Ingestor
from signaltools.core.signal import Signal
from signaltools.utils import ensure_non_negative_float, ensure_positive_int, to_1d_float_array


def test_bitlayer_empty_and_main(capsys: pytest.CaptureFixture[str]) -> None:
    assert bitlayer.bit_entropy([]) == 0.0
    assert bitlayer.run_lengths([]) == []
    assert bitlayer.bit_balance([]) == 0.0
    bitlayer.__main__()
    assert "BIT{" in capsys.readouterr().out


def test_bridge_pack_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    real_pack = struct.pack

    def fake_pack(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("signaltools.bridge.struct.pack", fake_pack)
    layered = analyze_signal_layered([0.1, -0.2, 0.3], source_type="numeric")
    assert layered.digital_layer["sample_count"] == 3
    monkeypatch.setattr("signaltools.bridge.struct.pack", real_pack)


def test_feature_zero_variance_cases() -> None:
    frame = [0.0, 0.0, 0.0, 0.0]
    features = st.frame_feature_vector(frame)
    assert features["dynamic_range_db"] == 0.0
    assert features["skewness"] == 0.0
    assert features["kurtosis"] == 0.0


def test_fingerprint_empty_signal_branch() -> None:
    fp = fingerprint_engine([], sample_rate=1000, frame_size=64, hop_size=32)
    assert len(fp.vector) == len(fp.labels)


def test_window_values_and_standardize_zero() -> None:
    assert len(_window_values(4, "rect")) == 4
    assert len(_window_values(4, "hann")) == 4
    assert len(_window_values(4, "hamming")) == 4
    assert len(_window_values(4, "blackman")) == 4
    with pytest.raises(ValueError):
        _window_values(4, "weird")
    assert st.standardize_signal([1.0, 1.0]) == [0.0, 0.0]


def test_io_additional_branches(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav_path = tmp_path / "s.wav"
    st.write_wav(wav_path, [0.0, 0.1, -0.1], sample_rate=8000)
    assert Ingestor.from_wav(str(wav_path)).shape[0] == 3

    empty_txt = tmp_path / "empty.txt"
    empty_txt.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        Ingestor.from_text(str(empty_txt))

    list_json = tmp_path / "list.json"
    list_json.write_text("[1,2,3]", encoding="utf-8")
    assert Ingestor.from_json(str(list_json)).tolist() == [1.0, 2.0, 3.0]

    key_json = tmp_path / "key.json"
    key_json.write_text('{"samples":[4,5]}', encoding="utf-8")
    assert Ingestor.from_json(str(key_json)).tolist() == [4.0, 5.0]


def test_utils_validation_branches() -> None:
    assert ensure_positive_int(1, "x") == 1
    assert ensure_non_negative_float(0, "x") == 0.0
    with pytest.raises(Exception):
        ensure_positive_int(0, "x")
    with pytest.raises(Exception):
        ensure_non_negative_float(-1, "x")
    with pytest.raises(Exception):
        ensure_non_negative_float("bad", "x")
    with pytest.raises(Exception):
        to_1d_float_array(np.array([[1.0, 2.0]]))
    with pytest.raises(Exception):
        to_1d_float_array([1.0, float("nan")])


def test_filters_and_spectral_more_edges() -> None:
    assert st.median_filter([1, 2, 3], 4)
    assert st.normalize_peak([0.0, 0.0]) == [0.0, 0.0]
    assert st.frequency_axis(8, 1000, real_only=False)
    assert st.power_spectrum([0.0, 0.0, 0.0]) == [0.0, 0.0]
    assert st.spectral_rolloff([0.0, 0.0], sample_rate=1000) == 0.0
    assert st.spectral_centroid([0.0, 0.0], sample_rate=1000) == 0.0
    assert st.spectral_bandwidth([0.0, 0.0], sample_rate=1000) == 0.0
    assert st.estimate_pitch([0.0], sample_rate=1000) == 0.0


def test_detect_additional_edges() -> None:
    assert st.adaptive_threshold([]) == 0.0
    assert st.local_peaks([1.0, 1.0], min_height=0.0, min_distance=1) == []


def test_core_signal_cache_and_glyph_vector() -> None:
    signal = Signal(np.array([0.1, -0.2, 0.3, -0.4]))
    signal.extract_features()
    assert signal._get_spectral()["magnitude"]
    assert len(signal.get_glyph_vector()) == 6


def test_remaining_branches(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert st.spectral_flatness([0.0, 0.0]) == 0.0
    assert st.band_energy([1.0, 2.0, 3.0], 1000, 9000, 10000) == 0.0
    assert st.stft([1.0, 2.0, 3.0, 4.0], frame_size=2, hop_size=1, window="hamming")
    assert st.stft([1.0, 2.0, 3.0, 4.0], frame_size=2, hop_size=1, window="blackman")
    assert st.stft([1.0, 2.0, 3.0, 4.0], frame_size=2, hop_size=1, window="rect")
    assert st.fft_bandpass([], 1000, 1, 2) == []
    from signaltools.framing import _window_values
    assert _window_values(0, "rect").size == 0
    assert st.frame_signal([1.0, 2.0], st.FrameConfig(frame_size=4, hop_size=1, pad_end=False)) == []

    class DummyCompleted:
        def __init__(self, stdout: bytes):
            self.stdout = stdout

    captured = {}

    def fake_run(cmd, capture_output, check):
        captured["cmd"] = cmd
        return DummyCompleted(b"1234")

    audio_path = tmp_path / "x.wav"
    audio_path.write_bytes(b"RIFF")
    monkeypatch.setattr("signaltools.io.subprocess.run", fake_run)
    st.read_audio_file(audio_path, filters="highpass=f=100")
    assert "-af" in captured["cmd"]

    with pytest.raises(Exception):
        to_1d_float_array([object()])
