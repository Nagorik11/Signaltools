from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from signaltools import (
    build_layer_alpha_masks,
    decompose_image_layers,
    export_comparison_mosaic,
    forensic_decompose_image,
    save_alpha_masks,
)


def test_ink_masks_alpha_and_mosaic(tmp_path: Path) -> None:
    image = np.full((12, 12), 170.0, dtype=float)
    image[5:8, 5:8] = 20.0
    image[2, 2] = 255.0
    result = decompose_image_layers(
        image,
        background_kernel_size=5,
        background_smooth_kernel_size=3,
        reflection_percentile=98.0,
    )
    ink = np.asarray(result.ink_mask, dtype=float)
    stroke = np.asarray(result.stroke_mask, dtype=float)
    assert ink.shape == image.shape
    assert stroke.shape == image.shape
    assert np.max(ink) >= 1.0

    alpha = build_layer_alpha_masks(result)
    assert "foreground" in alpha
    saved_alpha = save_alpha_masks(result, tmp_path / "alpha")
    assert Path(saved_alpha.paths["foreground"]).exists()

    mosaic = export_comparison_mosaic(result, tmp_path / "mosaic.png")
    assert Path(mosaic.output_path).exists()
    assert "ink_mask" in mosaic.tile_names


def test_forensic_image_bundle_with_alpha_and_mosaic(tmp_path: Path) -> None:
    image = np.full((16, 16), 160, dtype=np.uint8)
    image[6:10, 6:10] = 25
    image[3, 3] = 255
    path = tmp_path / "input.png"
    Image.fromarray(image, mode="L").save(path)

    result = forensic_decompose_image(
        path,
        case_id="IMG-2",
        examiner="FVE",
        signer="FVE",
        signing_key="k",
        output_dir=tmp_path / "bundle",
        background_kernel_size=5,
        background_smooth_kernel_size=3,
        reflection_percentile=98.0,
    )
    assert result.saved_alpha_masks is not None
    assert result.mosaic is not None
    assert result.bundle_paths is not None
    assert Path(result.bundle_paths.mosaic_path).exists()
    assert Path(result.bundle_paths.alpha_masks_dir).exists()
    assert "foreground" in result.alpha_hashes
