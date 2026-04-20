from __future__ import annotations

import builtins
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import signaltools as st
from signaltools.__main__ import _load_signal, main as cli_main
from signaltools.core.signal import Signal
from signaltools.detect import anomaly_score, onset_strength
from signaltools.features import (
    crest_factor,
    dynamic_range,
    kurtosis,
    mean,
    median,
    median_abs_deviation,
    peak_to_peak,
    signal_energy,
    skewness,
    stddev,
    variance,
    waveform_length,
    zero_crossing_rate,
)
from signaltools.io.ingestor import Ingestor as LegacyIngestor
from signaltools.logging_utils import DEFAULT_LOG_FORMAT, configure_logging
from signaltools.test import generate_demo_signal, main as demo_main
from signaltools.utils import round_float, safe_mean, to_1d_float_array


def test_demo_script_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert len(generate_demo_signal()) > 0
    assert demo_main() == 0
    assert (tmp_path / "output" / "advanced_analysis.json").exists()


def test_cli_loaders_for_wav_text_and_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav_path = tmp_path / "a.wav"
    txt_path = tmp_path / "a.txt"
    bin_path = tmp_path / "a.bin"
    st.write_wav(wav_path, [0.1, -0.1, 0.2, -0.2], sample_rate=8000)
    txt_path.write_text("AB", encoding="utf-8")
    bin_path.write_bytes(b"\x01\x02\x03")
    assert len(_load_signal(wav_path)) == 4
    assert _load_signal(txt_path) == [65.0, 66.0]
    assert _load_signal(bin_path) == [1.0, 2.0, 3.0]
    monkeypatch.setattr("sys.argv", ["signaltools", str(bin_path)])
    assert cli_main() == 0


def test_logging_helpers() -> None:
    configure_logging("INFO")
    logger = st.get_logger("signaltools.edge")
    assert logger.name == "signaltools.edge"
    assert "%(message)s" in DEFAULT_LOG_FORMAT


def test_utils_and_feature_empty_cases() -> None:
    assert safe_mean([]) == 0.0
    assert round_float(1.23456789) == 1.234568
    assert to_1d_float_array([]).size == 0
    assert mean([]) == 0.0
    assert variance([]) == 0.0
    assert stddev([]) == 0.0
    assert median([]) == 0.0
    assert median_abs_deviation([]) == 0.0
    assert zero_crossing_rate([1.0]) == 0.0
    assert peak_to_peak([]) == 0.0
    assert crest_factor([]) == 0.0
    assert signal_energy([]) == 0.0
    assert waveform_length([1.0]) == 0.0
    assert dynamic_range([]) == 0.0
    assert skewness([1.0]) == 0.0
    assert kurtosis([1.0]) == 0.0
    assert anomaly_score([]) == []
    assert anomaly_score([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]
    assert onset_strength([1.0]) == []


def test_framing_and_filters_edge_cases() -> None:
    assert st.normalize_signal([]) == []
    assert st.detrend_mean([]) == []
    assert st.standardize_signal([]) == []
    assert st.moving_average([], 3) == []
    assert st.median_filter([], 5) == []
    assert st.remove_dc([]) == []
    assert st.normalize_peak([]) == []
    with pytest.raises(ValueError):
        st.fft_bandpass([1, 2, 3], 1000, 20, 10)


def test_spectral_edge_cases() -> None:
    assert st.dft([]) == {"real": [], "imag": [], "magnitude": [], "phase": []}
    assert st.frequency_axis(0, 1000) == []
    assert st.power_spectrum([]) == []
    assert st.spectral_centroid([], sample_rate=1000) == 0.0
    assert st.spectral_bandwidth([], sample_rate=1000) == 0.0
    assert st.spectral_rolloff([], sample_rate=1000) == 0.0
    assert st.band_energy([], 1000, 1, 2) == 0.0
    assert st.band_energy([1, 2, 3], 1000, 5, 5) == 0.0
    assert st.stft([], frame_size=16, hop_size=8) == []
    assert st.spectrogram_matrix([], frame_size=16, hop_size=8) == []
    assert st.autocorrelation([]) == []
    assert st.estimate_pitch([], sample_rate=1000) == 0.0


def test_detection_edge_cases() -> None:
    assert st.adaptive_threshold([]) == 0.0
    assert st.local_peaks([1.0, 2.0], min_height=0.0, min_distance=1) == []


def test_bridge_bytes_and_core_glyph_vector() -> None:
    layered = st.analyze_signal_layered(b"\x00\x01\x02", source_type="bytes")
    assert layered.digital_layer["data_type"] == "raw_bytes"
    signal = Signal([0.1, -0.2, 0.3, -0.4])
    glyph = signal.get_glyph_vector()
    assert len(glyph) == 6


def test_legacy_ingestor_module_and_json_key_path(tmp_path: Path) -> None:
    json_path = tmp_path / "sig.json"
    json_path.write_text(json.dumps({"samples": [1, 2, 3]}), encoding="utf-8")
    assert LegacyIngestor.from_json(str(json_path)).tolist() == [1.0, 2.0, 3.0]


def test_read_audio_file_subprocess_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "x.wav"
    audio_path.write_bytes(b"RIFF")

    class DummyCompleted:
        def __init__(self, stdout: bytes):
            self.stdout = stdout

    def fake_ok(*args, **kwargs):
        return DummyCompleted(b"\x00\x01")

    monkeypatch.setattr("signaltools.io.subprocess.run", fake_ok)
    buffer = st.read_audio_file(audio_path)
    assert buffer.raw == b"\x00\x01"

    def fake_called(*args, **kwargs):
        raise importlib.import_module("subprocess").CalledProcessError(1, "ffmpeg", stderr=b"boom")

    monkeypatch.setattr("signaltools.io.subprocess.run", fake_called)
    with pytest.raises(RuntimeError):
        st.read_audio_file(audio_path)

    def fake_missing(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr("signaltools.io.subprocess.run", fake_missing)
    with pytest.raises(RuntimeError):
        st.read_audio_file(audio_path)


def test_ingestor_pcap_and_video_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class DummyPacket:
        def __init__(self, size: int, timestamp: float):
            self._size = size
            self.time = timestamp

        def __len__(self) -> int:
            return self._size

    fake_scapy_all = types.SimpleNamespace(rdpcap=lambda _: [DummyPacket(10, 1.0), DummyPacket(20, 1.5)])
    monkeypatch.setitem(sys.modules, "scapy", types.SimpleNamespace(all=fake_scapy_all))
    monkeypatch.setitem(sys.modules, "scapy.all", fake_scapy_all)
    assert LegacyIngestor.from_pcap(str(tmp_path / "dummy.pcap"), feature="size").tolist() == [10.0, 20.0]
    assert LegacyIngestor.from_pcap(str(tmp_path / "dummy.pcap"), feature="time").tolist() == [0.5]
    with pytest.raises(ValueError):
        LegacyIngestor.from_pcap(str(tmp_path / "dummy.pcap"), feature="other")

    fake_empty_scapy_all = types.SimpleNamespace(rdpcap=lambda _: [])
    monkeypatch.setitem(sys.modules, "scapy.all", fake_empty_scapy_all)
    with pytest.raises(ValueError):
        LegacyIngestor.from_pcap(str(tmp_path / "dummy.pcap"), feature="size")

    monkeypatch.delitem(sys.modules, "scapy.all", raising=False)
    monkeypatch.delitem(sys.modules, "scapy", raising=False)
    with pytest.raises(ImportError):
        LegacyIngestor.from_pcap(str(tmp_path / "dummy.pcap"), feature="size")

    class DummyCapture:
        def __init__(self, frames):
            self.frames = iter(frames)

        def isOpened(self):
            return True

        def read(self):
            try:
                return True, next(self.frames)
            except StopIteration:
                return False, None

        def release(self):
            return None

    fake_cv2 = types.SimpleNamespace(
        VideoCapture=lambda _: DummyCapture([np.ones((2, 2, 3)), np.ones((2, 2, 3)) * 2]),
        COLOR_BGR2GRAY=0,
        cvtColor=lambda frame, _: frame[:, :, 0],
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    video_signal = LegacyIngestor.from_video(str(tmp_path / "dummy.mp4"))
    assert video_signal.tolist() == [1.0, 2.0]

    fake_cv2_empty = types.SimpleNamespace(
        VideoCapture=lambda _: DummyCapture([]),
        COLOR_BGR2GRAY=0,
        cvtColor=lambda frame, _: frame,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2_empty)
    with pytest.raises(ValueError):
        LegacyIngestor.from_video(str(tmp_path / "dummy.mp4"))

    monkeypatch.delitem(sys.modules, "cv2", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("missing cv2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        LegacyIngestor.from_video(str(tmp_path / "dummy.mp4"))
