from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

import signaltools as st
from signaltools.__main__ import main as cli_main
from signaltools.core.analyzer import SignalAnalyzer
from signaltools.core.signal import Signal
from signaltools.io import Ingestor
from signaltools.io.wav import load_wav, save_wav
from signaltools.manager import Manager


@pytest.fixture
def tmp_signal_file(tmp_path: Path) -> Path:
    path = tmp_path / "signal.bin"
    path.write_bytes(b"\x00\x01\x02\x03\x04\x05")
    return path


def test_signal_buffer_and_numeric_views(tmp_signal_file: Path) -> None:
    buffer = st.read_signal_file(tmp_signal_file)
    assert buffer.size == 6
    assert buffer.hex_preview(3) == "00 01 02"
    views = st.guess_numeric_views(buffer.raw)
    assert views["uint8"][:3] == [0, 1, 2]


def test_ingestor_from_json_and_text(tmp_path: Path) -> None:
    json_path = tmp_path / "sig.json"
    json_path.write_text(json.dumps({"signal": [1, 2, 3]}), encoding="utf-8")
    txt_path = tmp_path / "sig.txt"
    txt_path.write_text("ABC", encoding="utf-8")
    assert Ingestor.from_json(str(json_path)).tolist() == [1.0, 2.0, 3.0]
    assert Ingestor.from_text(str(txt_path)).tolist() == [65.0, 66.0, 67.0]


def test_ingestor_from_json_invalid(tmp_path: Path) -> None:
    json_path = tmp_path / "bad.json"
    json_path.write_text(json.dumps({"other": [1, 2]}), encoding="utf-8")
    with pytest.raises(ValueError):
        Ingestor.from_json(str(json_path))


def test_read_audio_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        st.read_audio_file(tmp_path / "missing.wav")


def test_wav_helpers_roundtrip(tmp_path: Path) -> None:
    signal = np.array([0.0, 0.25, -0.25, 0.5], dtype=np.float32)
    wav_path = tmp_path / "sample.wav"
    save_wav(signal, str(wav_path), sample_rate=8000)
    loaded = load_wav(str(wav_path))
    assert loaded.shape[0] == 4
    assert float(np.max(np.abs(loaded))) <= 1.0


def test_core_signal_and_analyzer() -> None:
    signal = Signal([0.1, -0.2, 0.3, -0.4] * 100).normalize()
    features = signal.extract_features()
    assert "rms" in features
    assert signal.energy > 0
    analyzer = SignalAnalyzer(signal, window_size=40)
    summary = analyzer.generate_summary()
    assert summary["total_windows"] > 0


def test_manager_summary() -> None:
    manager = Manager([0.1, -0.2, 0.3, -0.4] * 100)
    summary = manager.summary()
    assert isinstance(summary, dict)


def test_legacy_io_module_exports(tmp_path: Path) -> None:
    import signaltools.io as package_io
    import signaltools.io as alias_check
    import signaltools.io as _  # deliberate import exercise
    import signaltools.io as same
    import signaltools.io as again
    import signaltools.io as duplicate
    import signaltools.io as duplicate2
    import signaltools.io as duplicate3
    import signaltools.io as duplicate4
    import signaltools.io as duplicate5
    import signaltools.io as duplicate6
    import signaltools.io as duplicate7
    import signaltools.io as duplicate8
    import signaltools.io as duplicate9
    import signaltools.io as duplicate10
    import signaltools.io as duplicate11
    import signaltools.io as duplicate12
    import signaltools.io as duplicate13
    import signaltools.io as duplicate14
    import signaltools.io as duplicate15
    import signaltools.io as duplicate16
    import signaltools.io as duplicate17
    import signaltools.io as duplicate18
    path = tmp_path / "sample.bin"
    path.write_bytes(struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))
    buffer = package_io.read_signal_file(path)
    assert alias_check.guess_numeric_views(buffer.raw)["float32_le"][0] == 1.0
    assert same.SignalBuffer is duplicate.SignalBuffer
    assert again.Ingestor is duplicate2.Ingestor
    assert duplicate3.read_wav is duplicate4.read_wav
    assert duplicate5.write_wav is duplicate6.write_wav
    assert duplicate7.decode_uint8 is duplicate8.decode_uint8
    assert duplicate9.decode_int16_le is duplicate10.decode_int16_le
    assert duplicate11.decode_uint16_le is duplicate12.decode_uint16_le
    assert duplicate13.decode_float32_le is duplicate14.decode_float32_le
    assert duplicate15.read_signal_file is duplicate16.read_signal_file
    assert duplicate17.read_audio_file is duplicate18.read_audio_file


def test_cli_main_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    signal_path = tmp_path / "signal.json"
    signal_path.write_text(json.dumps({"signal": [0.1, 0.2, -0.1, 0.4] * 100}), encoding="utf-8")
    output_path = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        ["signaltools", str(signal_path), "--sample-rate", "1000", "--output", str(output_path)],
    )
    code = cli_main()
    assert code == 0
    assert output_path.exists()


def test_cli_main_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setattr("sys.argv", ["signaltools", str(missing)])
    code = cli_main()
    assert code == 1
