"""Forensic pipeline helpers for image decomposition bundles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .forensics import (
    ChainOfCustody,
    ReportSignature,
    TimestampSeal,
    append_chain_of_custody_event,
    create_chain_of_custody,
    create_evidence_manifest,
    create_timestamp_seal,
    sign_report,
)
from .image_decomposition import (
    ImageLayerDecomposition,
    SavedLayerImages,
    decompose_image_layers,
    save_alpha_masks,
    save_decomposition_layers,
)
from .image_visualization import ComparisonMosaicResult, export_comparison_mosaic


@dataclass
class ForensicImageBundlePaths:
    bundle_dir: str
    report_json: str
    manifest_json: str
    chain_of_custody_json: str
    layer_hashes_json: str
    decomposition_json: str
    layers_dir: str
    alpha_masks_dir: str
    mosaic_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ForensicImageAnalysisResult:
    manifest: Any
    decomposition: ImageLayerDecomposition
    layer_hashes: dict[str, dict[str, str]]
    alpha_hashes: dict[str, dict[str, str]]
    timestamp_seal: TimestampSeal
    report_signature: ReportSignature
    chain_of_custody: ChainOfCustody
    saved_layers: SavedLayerImages | None
    saved_alpha_masks: SavedLayerImages | None
    mosaic: ComparisonMosaicResult | None
    bundle_paths: ForensicImageBundlePaths | None
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "decomposition": self.decomposition.to_dict(),
            "layer_hashes": self.layer_hashes,
            "alpha_hashes": self.alpha_hashes,
            "timestamp_seal": self.timestamp_seal.to_dict(),
            "report_signature": self.report_signature.to_dict(),
            "chain_of_custody": self.chain_of_custody.to_dict(),
            "saved_layers": self.saved_layers.to_dict() if self.saved_layers else None,
            "saved_alpha_masks": self.saved_alpha_masks.to_dict() if self.saved_alpha_masks else None,
            "mosaic": self.mosaic.to_dict() if self.mosaic else None,
            "bundle_paths": self.bundle_paths.to_dict() if self.bundle_paths else None,
            "meta": self.meta,
        }


def _load_image(path: str | Path) -> np.ndarray:
    image = Image.open(path)
    return np.asarray(image, dtype=np.float64)


def _hash_file(path: str | Path) -> dict[str, str]:
    from .forensics import hash_file

    return hash_file(path).to_dict()


def forensic_decompose_image(
    source: str | Path | np.ndarray,
    *,
    case_id: str = "",
    examiner: str = "",
    notes: str = "",
    custody_location: str = "image-lab-intake",
    signer: str = "",
    signing_key: str | None = None,
    output_dir: str | Path | None = None,
    save_layers: bool = True,
    save_alpha: bool = True,
    export_mosaic: bool = True,
    layer_prefix: str = "image_layer",
    alpha_prefix: str = "alpha",
    **decomposition_kwargs: Any,
) -> ForensicImageAnalysisResult:
    if isinstance(source, (str, Path)):
        image = _load_image(source)
    else:
        image = np.asarray(source, dtype=np.float64)

    manifest = create_evidence_manifest(source if isinstance(source, (str, Path)) else image, case_id=case_id, examiner=examiner, notes=notes)
    decomposition = decompose_image_layers(image, **decomposition_kwargs)

    chain = create_chain_of_custody(
        case_id=case_id or "UNSPECIFIED",
        evidence_sha256=manifest.hashes.sha256,
        actor=examiner or "unknown_examiner",
        location=custody_location,
        notes=notes or "Image evidence acquired for decomposition.",
    )
    append_chain_of_custody_event(
        chain,
        actor=examiner or "unknown_examiner",
        action="image_decomposed",
        location=custody_location,
        notes="Generated image layers and forensic metadata.",
    )

    saved_layers = None
    saved_alpha_masks = None
    mosaic = None
    layer_hashes: dict[str, dict[str, str]] = {}
    alpha_hashes: dict[str, dict[str, str]] = {}
    bundle_paths = None

    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        layers_dir = root / "layers"
        alpha_dir = root / "alpha_masks"
        mosaic_path = root / "comparison_mosaic.png"

        if save_layers:
            saved_layers = save_decomposition_layers(decomposition, layers_dir, prefix=layer_prefix)
            for name, path in saved_layers.paths.items():
                layer_hashes[name] = _hash_file(path)
        else:
            layers_dir.mkdir(parents=True, exist_ok=True)

        if save_alpha:
            saved_alpha_masks = save_alpha_masks(decomposition, alpha_dir, prefix=alpha_prefix)
            for name, path in saved_alpha_masks.paths.items():
                alpha_hashes[name] = _hash_file(path)
        else:
            alpha_dir.mkdir(parents=True, exist_ok=True)

        if export_mosaic:
            mosaic = export_comparison_mosaic(decomposition, mosaic_path)
        else:
            mosaic_path.parent.mkdir(parents=True, exist_ok=True)

        manifest_path = root / "manifest.json"
        decomposition_path = root / "decomposition.json"
        layer_hashes_path = root / "layer_hashes.json"
        custody_path = root / "chain_of_custody.json"
        report_path = root / "forensic_image_report.json"

        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        decomposition_path.write_text(json.dumps(decomposition.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        layer_hashes_path.write_text(
            json.dumps({"layers": layer_hashes, "alpha_masks": alpha_hashes}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        custody_path.write_text(json.dumps(chain.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

        bundle_paths = ForensicImageBundlePaths(
            bundle_dir=str(root),
            report_json=str(report_path),
            manifest_json=str(manifest_path),
            chain_of_custody_json=str(custody_path),
            layer_hashes_json=str(layer_hashes_path),
            decomposition_json=str(decomposition_path),
            layers_dir=str(layers_dir),
            alpha_masks_dir=str(alpha_dir),
            mosaic_path=str(mosaic_path),
        )
    else:
        layer_hashes = {}
        alpha_hashes = {}

    report_core = {
        "manifest": manifest.to_dict(),
        "decomposition_meta": decomposition.meta,
        "chain_of_custody": chain.to_dict(),
        "layer_hashes": layer_hashes,
        "alpha_hashes": alpha_hashes,
    }
    timestamp_seal = create_timestamp_seal(report_core)
    report_signature = sign_report(
        {
            **report_core,
            "timestamp_seal": timestamp_seal.to_dict(),
        },
        signer=signer or examiner,
        signing_key=signing_key,
    )
    append_chain_of_custody_event(
        chain,
        actor=examiner or "unknown_examiner",
        action="report_signed",
        location=custody_location,
        item_sha256=report_signature.scope_sha256,
        notes=f"algorithm={report_signature.algorithm}",
    )

    result = ForensicImageAnalysisResult(
        manifest=manifest,
        decomposition=decomposition,
        layer_hashes=layer_hashes,
        alpha_hashes=alpha_hashes,
        timestamp_seal=timestamp_seal,
        report_signature=report_signature,
        chain_of_custody=chain,
        saved_layers=saved_layers,
        saved_alpha_masks=saved_alpha_masks,
        mosaic=mosaic,
        bundle_paths=bundle_paths,
        meta={
            "save_layers": save_layers,
            "save_alpha": save_alpha,
            "export_mosaic": export_mosaic,
            "layer_prefix": layer_prefix,
            "alpha_prefix": alpha_prefix,
            "input_shape": list(image.shape),
        },
    )

    if bundle_paths is not None:
        Path(bundle_paths.report_json).write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return result


__all__ = [
    "ForensicImageAnalysisResult",
    "ForensicImageBundlePaths",
    "forensic_decompose_image",
]
