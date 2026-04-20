from __future__ import annotations

import json
from pathlib import Path

from signaltools import (
    create_evidence_manifest,
    forensic_analyze_signal,
    get_forensic_profile,
    hash_bytes,
)
from signaltools.__main__ import main


def test_hash_and_manifest(tmp_path: Path) -> None:
    payload = b"abc123"
    hashes = hash_bytes(payload)
    assert len(hashes.sha256) == 64
    assert len(hashes.sha1) == 40
    assert len(hashes.md5) == 32

    path = tmp_path / "evidence.bin"
    path.write_bytes(payload)
    manifest = create_evidence_manifest(path, case_id="CASE-1", examiner="Analyst")
    assert manifest.source_name == "evidence.bin"
    assert manifest.case_id == "CASE-1"
    assert manifest.hashes.sha256 == hashes.sha256


def test_forensic_analyze_signal_bundle(tmp_path: Path) -> None:
    sample = tmp_path / "signal.json"
    sample.write_text(json.dumps([0.0, 1.0, 0.5, -0.25, 0.75, -0.1, 0.2, 0.0]), encoding="utf-8")

    result = forensic_analyze_signal(
        sample,
        profile="network_telemetry",
        case_id="NET-42",
        examiner="FVE",
        output_dir=tmp_path / "bundle",
    )
    assert result.manifest.case_id == "NET-42"
    assert result.profile.name == "network_telemetry"
    assert len(result.audit_trail) == 2
    assert result.analysis.summary["samples"] == 8
    assert result.bundle_paths is not None
    for expected in [
        Path(result.bundle_paths.manifest_json),
        Path(result.bundle_paths.audit_json),
        Path(result.bundle_paths.analysis_json),
        Path(result.bundle_paths.report_json),
    ]:
        assert expected.exists()


def test_get_forensic_profile_and_cli(tmp_path: Path, monkeypatch) -> None:
    profile = get_forensic_profile("audio_voice")
    assert profile.sample_rate == 16000

    sample = tmp_path / "signal.json"
    sample.write_text(json.dumps([0.0, 0.25, 0.5, 0.25, 0.0, -0.25, -0.5, -0.25]), encoding="utf-8")
    output = tmp_path / "report.json"
    bundle_dir = tmp_path / "bundle"
    monkeypatch.setattr(
        "sys.argv",
        [
            "signaltools",
            str(sample),
            "--forensic",
            "--profile",
            "audio_voice",
            "--case-id",
            "VOICE-7",
            "--examiner",
            "Lab",
            "--bundle-dir",
            str(bundle_dir),
            "--output",
            str(output),
        ],
    )
    exit_code = main()
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["manifest"]["case_id"] == "VOICE-7"
    assert payload["profile"]["name"] == "audio_voice"
    assert (bundle_dir / "forensic_report.json").exists()
