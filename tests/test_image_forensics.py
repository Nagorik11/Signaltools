from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from signaltools import (
    decompose_image_layers,
    forensic_decompose_image,
    save_decomposition_layers,
)


def test_save_decomposition_layers(tmp_path: Path) -> None:
    image = np.full((8, 8), 180.0, dtype=float)
    image[4, 4] = 10.0
    image[1, 1] = 255.0
    result = decompose_image_layers(image, background_kernel_size=3, background_smooth_kernel_size=3)
    saved = save_decomposition_layers(result, tmp_path / "layers", prefix="demo")
    assert Path(saved.paths["foreground"]).exists()
    assert Path(saved.paths["illumination"]).exists()
    assert Path(saved.paths["specular"]).exists()


def test_forensic_decompose_image(tmp_path: Path) -> None:
    image = np.full((10, 10), 150, dtype=np.uint8)
    image[3, 3] = 20
    image[1, 1] = 255
    path = tmp_path / "input.png"
    Image.fromarray(image, mode="L").save(path)

    result = forensic_decompose_image(
        path,
        case_id="IMG-1",
        examiner="FVE",
        signer="FVE",
        signing_key="image-secret",
        output_dir=tmp_path / "bundle",
        background_kernel_size=3,
        background_smooth_kernel_size=3,
        reflection_percentile=95.0,
    )
    assert result.bundle_paths is not None
    assert result.report_signature.algorithm == "HMAC-SHA256"
    assert len(result.chain_of_custody.events) == 3
    assert Path(result.bundle_paths.report_json).exists()
    assert Path(result.bundle_paths.chain_of_custody_json).exists()
    assert Path(result.bundle_paths.layers_dir).exists()
    assert "foreground" in result.layer_hashes
