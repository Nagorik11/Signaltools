from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from . import analyze_signal_advanced, configure_logging, forensic_analyze_signal
from .exceptions import SignalToolsError
from .io import Ingestor, read_wav


def _load_signal(path: Path) -> list[float]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return Ingestor.from_json(str(path)).astype(np.float32).tolist()
    if suffix == ".wav":
        return read_wav(path)
    if suffix in {".txt", ".md", ".log"}:
        return Ingestor.from_text(str(path)).astype(np.float32).tolist()
    raw = path.read_bytes()
    return [float(x) for x in raw]


def main() -> int:
    parser = argparse.ArgumentParser(description="signaltools advanced analysis CLI")
    parser.add_argument("input", help="Path to the input signal file (.json, .wav, .txt, or binary)")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--frame-size", type=int, default=256)
    parser.add_argument("--hop-size", type=int, default=128)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--log-level", type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--forensic", action="store_true", help="Generate a forensic-ready report with manifest and audit trail")
    parser.add_argument("--profile", type=str, default="generic_signal", help="Forensic profile to use when --forensic is enabled")
    parser.add_argument("--case-id", type=str, default="")
    parser.add_argument("--examiner", type=str, default="")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--bundle-dir", type=str, default="", help="Directory where forensic bundle files are written")
    parser.add_argument("--custody-location", type=str, default="lab-intake")
    parser.add_argument("--signer", type=str, default="")
    parser.add_argument("--signing-key", type=str, default="")
    args = parser.parse_args()

    configure_logging(args.log_level)

    try:
        input_path = Path(args.input)
        if args.forensic:
            result = forensic_analyze_signal(
                input_path,
                profile=args.profile,
                sample_rate=args.sample_rate,
                frame_size=args.frame_size,
                hop_size=args.hop_size,
                case_id=args.case_id,
                examiner=args.examiner,
                notes=args.notes,
                output_dir=args.bundle_dir or None,
                custody_location=args.custody_location,
                signer=args.signer,
                signing_key=args.signing_key or None,
            )
            payload = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        else:
            signal = _load_signal(input_path)
            analysis = analyze_signal_advanced(signal, sample_rate=args.sample_rate, frame_size=args.frame_size, hop_size=args.hop_size).to_dict()
            payload = json.dumps(analysis, indent=2, ensure_ascii=False)
    except (OSError, ValueError, SignalToolsError) as exc:
        print(f"Error: {exc}")
        return 1

    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print(f"Saved analysis to {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
