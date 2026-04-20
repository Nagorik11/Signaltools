from __future__ import annotations

import json
from pathlib import Path

from signaltools import (
    append_chain_of_custody_event,
    create_chain_of_custody,
    create_timestamp_seal,
    forensic_analyze_signal,
    get_forensic_profile,
    sign_report,
)


def test_timestamp_seal_and_signature() -> None:
    payload = {"a": 1, "b": [1, 2, 3]}
    seal = create_timestamp_seal(payload)
    assert len(seal.seal_sha256) == 64
    assert seal.trusted is False

    digest_sig = sign_report(payload, signer="Analyst")
    keyed_sig = sign_report(payload, signer="Analyst", signing_key="secret-key")
    assert digest_sig.algorithm == "SHA256-DIGEST"
    assert keyed_sig.algorithm == "HMAC-SHA256"
    assert keyed_sig.is_keyed is True


def test_chain_of_custody_append() -> None:
    chain = create_chain_of_custody(case_id="CASE-X", evidence_sha256="ab" * 32, actor="Lab", location="Vault")
    append_chain_of_custody_event(chain, actor="Lab", action="transferred", location="Bench", notes="Moved to bench")
    assert chain.case_id == "CASE-X"
    assert len(chain.events) == 2
    assert chain.events[1].action == "transferred"


def test_domain_profiles_and_forensic_analysis_signed(tmp_path: Path) -> None:
    sample = tmp_path / "trace.json"
    sample.write_text(json.dumps([0.0, 1.0, 0.0, -1.0, 0.0, 0.5, -0.5, 0.0]), encoding="utf-8")

    profile = get_forensic_profile("network_forensics")
    assert profile.domain == "network"

    result = forensic_analyze_signal(
        sample,
        profile="network_forensics",
        case_id="NET-900",
        examiner="FVE",
        custody_location="rack-A",
        signer="FVE",
        signing_key="case-key",
        output_dir=tmp_path / "bundle",
    )
    assert result.chain_of_custody is not None
    assert len(result.chain_of_custody.events) == 3
    assert result.report_signature is not None
    assert result.report_signature.algorithm == "HMAC-SHA256"
    assert result.timestamp_seal is not None
    assert result.profile.domain == "network"
    assert (tmp_path / "bundle" / "chain_of_custody.json").exists()
