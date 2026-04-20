from __future__ import annotations

import numpy as np

from signaltools import (
    decompose_image_layers,
    estimate_background,
    extract_foreground,
    reconstruct_from_layers,
    rgb_to_gray,
    wavelet_subbands_2d,
)


def test_rgb_to_gray_and_background_foreground() -> None:
    rgb = np.zeros((5, 5, 3), dtype=float)
    rgb[..., 0] = 10.0
    rgb[..., 1] = 20.0
    rgb[..., 2] = 30.0
    gray = np.asarray(rgb_to_gray(rgb), dtype=float)
    assert gray.shape == (5, 5)
    assert np.allclose(gray, gray[0, 0])

    image = np.full((7, 7), 200.0, dtype=float)
    image[3, 3] = 10.0
    background = np.asarray(estimate_background(image, kernel_size=3, smooth_kernel_size=3), dtype=float)
    foreground = np.asarray(extract_foreground(image, background, mode="dark"), dtype=float)
    assert background.shape == image.shape
    assert foreground.shape == image.shape
    assert foreground[3, 3] > 0.0


def test_decompose_image_layers_and_reconstruction() -> None:
    image = np.full((9, 9), 180.0, dtype=float)
    image[4, 4] = 20.0
    image[1, 1] = 255.0
    result = decompose_image_layers(
        image,
        background_kernel_size=3,
        background_smooth_kernel_size=3,
        denoise_kernel_size=3,
        edge_kernel_size=3,
        reflection_percentile=95.0,
        foreground_mode="dark",
        wavelet_family="haar",
        wavelet_level=1,
    )
    gray = np.asarray(result.grayscale, dtype=float)
    fg = np.asarray(result.foreground, dtype=float)
    reflections = np.asarray(result.reflections, dtype=float)
    assert gray.shape == image.shape
    assert fg[4, 4] > 0.0
    assert reflections[1, 1] == 1.0
    assert "wavelet_subbands" in result.meta

    recon = np.asarray(
        reconstruct_from_layers(
            background=result.background,
            foreground=result.foreground,
            reflections=np.zeros_like(gray),
            noise=np.zeros_like(gray),
            foreground_sign=-1.0,
        ),
        dtype=float,
    )
    assert recon.shape == image.shape


def test_wavelet_subbands_2d() -> None:
    image = np.arange(64, dtype=float).reshape(8, 8)
    subbands = wavelet_subbands_2d(image, family="haar", level=1)
    ll = np.asarray(subbands.ll, dtype=float)
    lh = np.asarray(subbands.lh, dtype=float)
    assert ll.ndim == 2
    assert lh.ndim == 2
    assert subbands.family == "haar"
