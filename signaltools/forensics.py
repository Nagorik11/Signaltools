from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .io import Ingestor, read_wav
from .pipeline import AdvancedSignalAnalysis, analyze_signal_advanced


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _to_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            arr = np.asarray(value, dtype=np.complex128)
        else:
            arr = np.asarray(value, dtype=np.float64)
        return arr.tobytes()
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value)
        if np.iscomplexobj(arr):
            arr = arr.astype(np.complex128)
        else:
            arr = arr.astype(np.float64)
        return arr.tobytes()
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")


@dataclass
class EvidenceHashes:
    md5: str
    sha1: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceManifest:
    source_name: str
    source_type: str
    size_bytes: int
    hashes: EvidenceHashes
    created_utc: str
    case_id: str
    examiner: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_type": self.source_type,
            "size_bytes": self.size_bytes,
            "hashes": self.hashes.to_dict(),
            "created_utc": self.created_utc,
            "case_id": self.case_id,
            "examiner": self.examiner,
            "notes": self.notes,
        }


@dataclass
class AuditStep:
    name: str
    timestamp_utc: str
    parameters: dict[str, Any]
    input_sha256: str
    output_sha256: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ForensicProfile:
    name: str
    description: str
    domain: str
    sample_rate: int
    frame_size: int
    hop_size: int
    normalization: str
    preserve_intermediate_hashes: bool = True
    recommended_outputs: tuple[str, ...] = ("manifest", "audit_trail", "analysis", "report")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["recommended_outputs"] = list(self.recommended_outputs)
        return payload


@dataclass
class TimestampSeal:
    sealed_utc: str
    authority: str
    source_sha256: str
    nonce: str
    seal_sha256: str
    trusted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReportSignature:
    signer: str
    algorithm: str
    signature_hex: str
    signed_utc: str
    scope_sha256: str
    is_keyed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChainOfCustodyEvent:
    event_id: str
    timestamp_utc: str
    actor: str
    action: str
    location: str
    item_sha256: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChainOfCustody:
    case_id: str
    evidence_sha256: str
    created_utc: str
    events: list[ChainOfCustodyEvent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "evidence_sha256": self.evidence_sha256,
            "created_utc": self.created_utc,
            "events": [event.to_dict() for event in self.events],
        }


@dataclass
class ForensicBundlePaths:
    bundle_dir: str
    manifest_json: str
    audit_json: str
    analysis_json: str
    report_json: str
    custody_json: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ForensicAnalysisResult:
    manifest: EvidenceManifest
    profile: ForensicProfile
    audit_trail: list[AuditStep]
    analysis: AdvancedSignalAnalysis
    environment: dict[str, Any]
    timestamp_seal: TimestampSeal | None = None
    report_signature: ReportSignature | None = None
    chain_of_custody: ChainOfCustody | None = None
    bundle_paths: ForensicBundlePaths | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "profile": self.profile.to_dict(),
            "audit_trail": [step.to_dict() for step in self.audit_trail],
            "analysis": self.analysis.to_dict(),
            "environment": self.environment,
            "timestamp_seal": self.timestamp_seal.to_dict() if self.timestamp_seal else None,
            "report_signature": self.report_signature.to_dict() if self.report_signature else None,
            "chain_of_custody": self.chain_of_custody.to_dict() if self.chain_of_custody else None,
            "bundle_paths": self.bundle_paths.to_dict() if self.bundle_paths else None,
        }


FORENSIC_PROFILES: dict[str, ForensicProfile] = {
    "generic_signal": ForensicProfile(
        name="generic_signal",
        description="Balanced profile for generic forensic signal triage.",
        domain="generic",
        sample_rate=44100,
        frame_size=256,
        hop_size=128,
        normalization="detrend_mean+normalize_signal",
    ),
    "audio_voice": ForensicProfile(
        name="audio_voice",
        description="Speech-oriented forensic profile with shorter frames for transients.",
        domain="audio",
        sample_rate=16000,
        frame_size=320,
        hop_size=160,
        normalization="detrend_mean+normalize_signal",
    ),
    "network_telemetry": ForensicProfile(
        name="network_telemetry",
        description="Profile for packet-derived or telemetry-derived scalar series.",
        domain="network",
        sample_rate=1000,
        frame_size=128,
        hop_size=64,
        normalization="detrend_mean+normalize_signal",
    ),
    "multimodal_probe": ForensicProfile(
        name="multimodal_probe",
        description="Conservative profile for heterogeneous scalarized channels.",
        domain="multimodal",
        sample_rate=8000,
        frame_size=128,
        hop_size=64,
        normalization="detrend_mean+normalize_signal",
    ),
    "audio_forensics": ForensicProfile(
        name="audio_forensics",
        description="Pericial audio profile for voice/noise/event examination.",
        domain="audio",
        sample_rate=48000,
        frame_size=512,
        hop_size=256,
        normalization="detrend_mean+normalize_signal",
    ),
    "image_forensics_scalarized": ForensicProfile(
        name="image_forensics_scalarized",
        description="Scalarized profile for line scans, traces and extracted image intensity series.",
        domain="image",
        sample_rate=4000,
        frame_size=256,
        hop_size=128,
        normalization="detrend_mean+normalize_signal",
    ),
    "network_forensics": ForensicProfile(
        name="network_forensics",
        description="Pericial profile for packet timing, rates, telemetry and derived network traces.",
        domain="network",
        sample_rate=2000,
        frame_size=128,
        hop_size=64,
        normalization="detrend_mean+normalize_signal",
    ),
    "multimodal_forensics": ForensicProfile(
        name="multimodal_forensics",
        description="Pericial profile for synchronized mixed-domain scalarized evidence.",
        domain="multimodal",
        sample_rate=12000,
        frame_size=256,
        hop_size=128,
        normalization="detrend_mean+normalize_signal",
    ),
}


def hash_bytes(payload: bytes) -> EvidenceHashes:
    return EvidenceHashes(
        md5=hashlib.md5(payload).hexdigest(),
        sha1=hashlib.sha1(payload).hexdigest(),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def hash_file(path: str | os.PathLike[str]) -> EvidenceHashes:
    data = Path(path).read_bytes()
    return hash_bytes(data)


def create_evidence_manifest(
    source: str | os.PathLike[str] | bytes | list[float] | np.ndarray,
    *,
    case_id: str = "",
    examiner: str = "",
    notes: str = "",
) -> EvidenceManifest:
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        payload = path.read_bytes()
        return EvidenceManifest(
            source_name=path.name,
            source_type=path.suffix.lower().lstrip(".") or "binary",
            size_bytes=len(payload),
            hashes=hash_bytes(payload),
            created_utc=_utc_now_iso(),
            case_id=case_id,
            examiner=examiner,
            notes=notes,
        )

    payload = _to_bytes(source)
    source_type = "ndarray" if isinstance(source, np.ndarray) else "in_memory_signal"
    return EvidenceManifest(
        source_name="in_memory_signal",
        source_type=source_type,
        size_bytes=len(payload),
        hashes=hash_bytes(payload),
        created_utc=_utc_now_iso(),
        case_id=case_id,
        examiner=examiner,
        notes=notes,
    )


def get_forensic_profile(name: str) -> ForensicProfile:
    try:
        return FORENSIC_PROFILES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(FORENSIC_PROFILES))
        raise ValueError(f"Unknown forensic profile '{name}'. Valid profiles: {valid}") from exc


def _load_signal_for_forensics(source: str | os.PathLike[str] | list[float] | np.ndarray) -> list[float]:
    if isinstance(source, np.ndarray):
        return source.astype(np.float64).tolist()
    if isinstance(source, list):
        return [float(x) for x in source]
    path = Path(source)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return Ingestor.from_json(str(path)).astype(np.float64).tolist()
    if suffix == ".wav":
        return [float(x) for x in read_wav(path)]
    if suffix in {".txt", ".log", ".md"}:
        return Ingestor.from_text(str(path)).astype(np.float64).tolist()
    return [float(x) for x in path.read_bytes()]


def _environment_snapshot() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "analysis_utc": _utc_now_iso(),
    }


def _audit_step(name: str, input_obj: Any, output_obj: Any, parameters: dict[str, Any], summary: dict[str, Any]) -> AuditStep:
    return AuditStep(
        name=name,
        timestamp_utc=_utc_now_iso(),
        parameters=parameters,
        input_sha256=hash_bytes(_to_bytes(input_obj)).sha256,
        output_sha256=hash_bytes(_to_bytes(output_obj)).sha256,
        summary=summary,
    )


def create_timestamp_seal(
    source: Any,
    *,
    authority: str = "local_clock_untrusted",
    nonce: str | None = None,
    trusted: bool = False,
) -> TimestampSeal:
    source_sha256 = hash_bytes(_to_bytes(source)).sha256
    token_nonce = nonce or secrets.token_hex(16)
    sealed_utc = _utc_now_iso()
    seal_payload = f"{sealed_utc}|{authority}|{source_sha256}|{token_nonce}".encode("utf-8")
    return TimestampSeal(
        sealed_utc=sealed_utc,
        authority=authority,
        source_sha256=source_sha256,
        nonce=token_nonce,
        seal_sha256=hash_bytes(seal_payload).sha256,
        trusted=trusted,
    )


def sign_report(
    report_payload: dict[str, Any],
    *,
    signer: str = "",
    signing_key: str | None = None,
) -> ReportSignature:
    canonical = json.dumps(report_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    scope_sha256 = hash_bytes(canonical).sha256
    if signing_key:
        signature_hex = hmac.new(signing_key.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
        algorithm = "HMAC-SHA256"
        is_keyed = True
    else:
        signature_hex = hashlib.sha256(canonical).hexdigest()
        algorithm = "SHA256-DIGEST"
        is_keyed = False
    return ReportSignature(
        signer=signer,
        algorithm=algorithm,
        signature_hex=signature_hex,
        signed_utc=_utc_now_iso(),
        scope_sha256=scope_sha256,
        is_keyed=is_keyed,
    )


def create_chain_of_custody(*, case_id: str, evidence_sha256: str, actor: str, location: str, notes: str = "") -> ChainOfCustody:
    event = ChainOfCustodyEvent(
        event_id="COC-0001",
        timestamp_utc=_utc_now_iso(),
        actor=actor,
        action="acquired",
        location=location,
        item_sha256=evidence_sha256,
        notes=notes or "Initial acquisition record.",
    )
    return ChainOfCustody(
        case_id=case_id,
        evidence_sha256=evidence_sha256,
        created_utc=_utc_now_iso(),
        events=[event],
    )


def append_chain_of_custody_event(
    chain: ChainOfCustody,
    *,
    actor: str,
    action: str,
    location: str,
    item_sha256: str | None = None,
    notes: str = "",
) -> ChainOfCustody:
    event_id = f"COC-{len(chain.events)+1:04d}"
    chain.events.append(
        ChainOfCustodyEvent(
            event_id=event_id,
            timestamp_utc=_utc_now_iso(),
            actor=actor,
            action=action,
            location=location,
            item_sha256=item_sha256 or chain.evidence_sha256,
            notes=notes,
        )
    )
    return chain


def write_forensic_bundle(result: ForensicAnalysisResult, output_dir: str | os.PathLike[str]) -> ForensicBundlePaths:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "manifest.json"
    audit_path = root / "audit_trail.json"
    analysis_path = root / "analysis.json"
    report_path = root / "forensic_report.json"
    custody_path = root / "chain_of_custody.json"

    manifest_path.write_text(json.dumps(result.manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    audit_path.write_text(
        json.dumps([step.to_dict() for step in result.audit_trail], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    analysis_path.write_text(json.dumps(result.analysis.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    if result.chain_of_custody:
        custody_path.write_text(json.dumps(result.chain_of_custody.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        custody_path.write_text(json.dumps({}, indent=2), encoding="utf-8")

    bundle_paths = ForensicBundlePaths(
        bundle_dir=str(root),
        manifest_json=str(manifest_path),
        audit_json=str(audit_path),
        analysis_json=str(analysis_path),
        report_json=str(report_path),
        custody_json=str(custody_path),
    )
    result.bundle_paths = bundle_paths
    report_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return bundle_paths


def forensic_analyze_signal(
    source: str | os.PathLike[str] | list[float] | np.ndarray,
    *,
    profile: str = "generic_signal",
    sample_rate: int | None = None,
    frame_size: int | None = None,
    hop_size: int | None = None,
    case_id: str = "",
    examiner: str = "",
    notes: str = "",
    output_dir: str | os.PathLike[str] | None = None,
    custody_location: str = "lab-intake",
    signer: str = "",
    signing_key: str | None = None,
) -> ForensicAnalysisResult:
    forensic_profile = get_forensic_profile(profile)
    final_sample_rate = int(sample_rate or forensic_profile.sample_rate)
    final_frame_size = int(frame_size or forensic_profile.frame_size)
    final_hop_size = int(hop_size or forensic_profile.hop_size)

    manifest = create_evidence_manifest(source, case_id=case_id, examiner=examiner, notes=notes)
    raw_signal = _load_signal_for_forensics(source)
    prepared_signal = np.asarray(raw_signal, dtype=np.float64)
    analysis = analyze_signal_advanced(
        prepared_signal.tolist(),
        sample_rate=final_sample_rate,
        frame_size=final_frame_size,
        hop_size=final_hop_size,
    )

    audit = [
        _audit_step(
            "ingest_source",
            source if not isinstance(source, (str, os.PathLike)) else str(source),
            raw_signal,
            parameters={"profile": profile},
            summary={"samples": len(raw_signal), "source_type": manifest.source_type},
        ),
        _audit_step(
            "advanced_analysis",
            raw_signal,
            analysis.to_dict(),
            parameters={
                "sample_rate": final_sample_rate,
                "frame_size": final_frame_size,
                "hop_size": final_hop_size,
            },
            summary={
                "frame_count": analysis.summary.get("frame_count", 0),
                "adaptive_event_count": analysis.temporal.get("adaptive_event_count", 0),
                "pitch_hz": analysis.spectral.get("pitch_hz", 0.0),
            },
        ),
    ]

    chain = create_chain_of_custody(
        case_id=case_id or "UNSPECIFIED",
        evidence_sha256=manifest.hashes.sha256,
        actor=examiner or "unknown_examiner",
        location=custody_location,
        notes=notes,
    )
    append_chain_of_custody_event(
        chain,
        actor=examiner or "unknown_examiner",
        action="analyzed",
        location=custody_location,
        item_sha256=manifest.hashes.sha256,
        notes=f"Profile={forensic_profile.name}; sample_rate={final_sample_rate}",
    )

    environment = _environment_snapshot()
    timestamp_seal = create_timestamp_seal(analysis.to_dict())

    result = ForensicAnalysisResult(
        manifest=manifest,
        profile=ForensicProfile(
            name=forensic_profile.name,
            description=forensic_profile.description,
            domain=forensic_profile.domain,
            sample_rate=final_sample_rate,
            frame_size=final_frame_size,
            hop_size=final_hop_size,
            normalization=forensic_profile.normalization,
            preserve_intermediate_hashes=forensic_profile.preserve_intermediate_hashes,
            recommended_outputs=forensic_profile.recommended_outputs,
        ),
        audit_trail=audit,
        analysis=analysis,
        environment=environment,
        timestamp_seal=timestamp_seal,
        report_signature=None,
        chain_of_custody=chain,
        bundle_paths=None,
    )

    signature_payload = {
        "manifest": result.manifest.to_dict(),
        "profile": result.profile.to_dict(),
        "audit_trail": [step.to_dict() for step in result.audit_trail],
        "analysis": result.analysis.to_dict(),
        "environment": result.environment,
        "timestamp_seal": result.timestamp_seal.to_dict() if result.timestamp_seal else None,
        "chain_of_custody": result.chain_of_custody.to_dict() if result.chain_of_custody else None,
    }
    result.report_signature = sign_report(signature_payload, signer=signer or examiner, signing_key=signing_key)

    append_chain_of_custody_event(
        chain,
        actor=examiner or "unknown_examiner",
        action="sealed_and_signed",
        location=custody_location,
        item_sha256=result.report_signature.scope_sha256,
        notes=f"algorithm={result.report_signature.algorithm}",
    )

    if output_dir is not None:
        write_forensic_bundle(result, output_dir)
    return result


__all__ = [
    "AuditStep",
    "ChainOfCustody",
    "ChainOfCustodyEvent",
    "EvidenceHashes",
    "EvidenceManifest",
    "ForensicAnalysisResult",
    "ForensicBundlePaths",
    "ForensicProfile",
    "FORENSIC_PROFILES",
    "ReportSignature",
    "TimestampSeal",
    "append_chain_of_custody_event",
    "create_chain_of_custody",
    "create_evidence_manifest",
    "create_timestamp_seal",
    "forensic_analyze_signal",
    "get_forensic_profile",
    "hash_bytes",
    "hash_file",
    "sign_report",
    "write_forensic_bundle",
]
