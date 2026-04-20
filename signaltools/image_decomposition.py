"""Image layer decomposition helpers for grayscale/RGB forensic-style analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .image_morphology import median_filter_2d, morphological_gradient_2d, opening_2d
from .wavelet_packet_2d import wavelet_packet_2d_decompose


@dataclass
class ImageLayerDecomposition:
    original: list[list[float]] | list[list[list[float]]]
    grayscale: list[list[float]]
    illumination: list[list[float]]
    background: list[list[float]]
    foreground: list[list[float]]
    texture: list[list[float]]
    reflections: list[list[float]]
    shadows: list[list[float]]
    specular: list[list[float]]
    noise: list[list[float]]
    edges: list[list[float]]
    denoised: list[list[float]]
    ink_mask: list[list[float]]
    stroke_mask: list[list[float]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WaveletSubbands2D:
    ll: list[list[float]]
    lh: list[list[float]]
    hl: list[list[float]]
    hh: list[list[float]]
    family: str
    level: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SavedLayerImages:
    output_dir: str
    paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_image(image: list[list[float]] | list[list[list[float]]] | np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float64)
    if x.ndim not in {2, 3}:
        raise ValueError("image must be a 2D grayscale image or a 3D RGB-like image")
    if x.ndim == 3 and x.shape[2] < 3:
        raise ValueError("3D image must have at least 3 channels")
    return x


def _ensure_positive_odd(value: int, name: str) -> int:
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer")
    return value


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    x = np.asarray(array, dtype=np.float64)
    if x.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    x = x - np.min(x)
    peak = np.max(x)
    if peak <= 1e-12:
        return np.zeros_like(x, dtype=np.uint8)
    return np.clip((255.0 * x / peak).round(), 0, 255).astype(np.uint8)


def _otsu_threshold(image: np.ndarray) -> float:
    data = _normalize_to_uint8(image).ravel()
    hist = np.bincount(data, minlength=256).astype(np.float64)
    total = hist.sum()
    sum_total = np.dot(np.arange(256), hist)
    sum_bg = 0.0
    weight_bg = 0.0
    max_var = -1.0
    threshold = 0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_var:
            max_var = between
            threshold = t
    return float(threshold)


def rgb_to_gray(image: list[list[float]] | list[list[list[float]]] | np.ndarray) -> list[list[float]]:
    x = _as_image(image)
    if x.ndim == 2:
        return x.tolist()
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.tolist()


def mean_filter_2d(image: list[list[float]] | np.ndarray, kernel_size: int = 5, mode: str = "edge") -> list[list[float]]:
    kernel_size = _ensure_positive_odd(kernel_size, "kernel_size")
    x = np.asarray(image, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("image must be 2D")
    pad = kernel_size // 2
    padded = np.pad(x, ((pad, pad), (pad, pad)), mode=mode)
    out = np.zeros_like(x)
    area = float(kernel_size * kernel_size)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = np.sum(padded[i : i + kernel_size, j : j + kernel_size]) / area
    return out.tolist()


def estimate_background(
    gray: list[list[float]] | np.ndarray,
    *,
    kernel_size: int = 15,
    smooth_kernel_size: int = 5,
) -> list[list[float]]:
    kernel_size = _ensure_positive_odd(kernel_size, "kernel_size")
    smooth_kernel_size = _ensure_positive_odd(smooth_kernel_size, "smooth_kernel_size")
    opened = np.asarray(opening_2d(gray, kernel_size=kernel_size), dtype=np.float64)
    smoothed = np.asarray(mean_filter_2d(opened, kernel_size=smooth_kernel_size), dtype=np.float64)
    return smoothed.tolist()


def estimate_illumination(gray: list[list[float]] | np.ndarray, *, kernel_size: int = 31) -> list[list[float]]:
    kernel_size = _ensure_positive_odd(kernel_size, "kernel_size")
    return mean_filter_2d(gray, kernel_size=kernel_size)


def extract_foreground(
    gray: list[list[float]] | np.ndarray,
    background: list[list[float]] | np.ndarray,
    *,
    mode: str = "dark",
) -> list[list[float]]:
    x = np.asarray(gray, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)
    if x.shape != bg.shape:
        raise ValueError("gray and background must share the same shape")
    if mode == "dark":
        foreground = np.clip(bg - x, 0.0, None)
    elif mode == "bright":
        foreground = np.clip(x - bg, 0.0, None)
    else:
        raise ValueError("mode must be 'dark' or 'bright'")
    return foreground.tolist()


def detect_reflections(gray: list[list[float]] | np.ndarray, *, percentile: float = 98.0) -> list[list[float]]:
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be between 0 and 100")
    x = np.asarray(gray, dtype=np.float64)
    threshold = np.percentile(x, percentile)
    mask = (x >= threshold).astype(np.float64)
    return mask.tolist()


def decompose_shadows_specular(
    gray: list[list[float]] | np.ndarray,
    illumination: list[list[float]] | np.ndarray,
    *,
    shadow_strength: float = 0.08,
    specular_percentile: float = 99.0,
) -> tuple[list[list[float]], list[list[float]]]:
    if shadow_strength < 0:
        raise ValueError("shadow_strength must be non-negative")
    x = np.asarray(gray, dtype=np.float64)
    illum = np.asarray(illumination, dtype=np.float64)
    if x.shape != illum.shape:
        raise ValueError("gray and illumination must share the same shape")
    ratio = x / np.maximum(illum, 1e-9)
    shadow_map = np.clip(1.0 - ratio, 0.0, None)
    shadow_mask = (shadow_map >= shadow_strength).astype(np.float64) * shadow_map
    spec_threshold = np.percentile(ratio, specular_percentile)
    specular = np.clip(ratio - spec_threshold, 0.0, None)
    return shadow_mask.tolist(), specular.tolist()


def segment_ink_strokes(
    gray: list[list[float]] | np.ndarray,
    background: list[list[float]] | np.ndarray,
    *,
    method: str = "otsu",
    mode: str = "dark",
) -> tuple[list[list[float]], list[list[float]]]:
    foreground = np.asarray(extract_foreground(gray, background, mode=mode), dtype=np.float64)
    if method != "otsu":
        raise ValueError("method must be 'otsu'")
    threshold = _otsu_threshold(foreground)
    normalized = _normalize_to_uint8(foreground)
    ink_mask = (normalized >= threshold).astype(np.float64)
    stroke_strength = foreground * ink_mask
    return ink_mask.tolist(), stroke_strength.tolist()


def estimate_noise(gray: list[list[float]] | np.ndarray, denoised: list[list[float]] | np.ndarray) -> list[list[float]]:
    x = np.asarray(gray, dtype=np.float64)
    d = np.asarray(denoised, dtype=np.float64)
    if x.shape != d.shape:
        raise ValueError("gray and denoised must share the same shape")
    return (x - d).tolist()


def extract_texture(gray: list[list[float]] | np.ndarray, background: list[list[float]] | np.ndarray) -> list[list[float]]:
    x = np.asarray(gray, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)
    if x.shape != bg.shape:
        raise ValueError("gray and background must share the same shape")
    return (x - bg).tolist()


def simple_edges(gray: list[list[float]] | np.ndarray, *, kernel_size: int = 3) -> list[list[float]]:
    return morphological_gradient_2d(gray, kernel_size=kernel_size)


def denoise_image(gray: list[list[float]] | np.ndarray, *, kernel_size: int = 3) -> list[list[float]]:
    return median_filter_2d(gray, kernel_size=kernel_size)


def wavelet_subbands_2d(
    gray: list[list[float]] | np.ndarray,
    *,
    family: str = "haar",
    level: int = 1,
) -> WaveletSubbands2D:
    tree = wavelet_packet_2d_decompose(gray, level=level, family=family)
    ll = np.asarray(tree.nodes.get("ll", tree.nodes.get("")), dtype=np.float64)
    lh = np.asarray(tree.nodes.get("lh", np.zeros_like(ll)), dtype=np.float64)
    hl = np.asarray(tree.nodes.get("hl", np.zeros_like(ll)), dtype=np.float64)
    hh = np.asarray(tree.nodes.get("hh", np.zeros_like(ll)), dtype=np.float64)
    return WaveletSubbands2D(
        ll=ll.tolist(),
        lh=lh.tolist(),
        hl=hl.tolist(),
        hh=hh.tolist(),
        family=family,
        level=level,
    )


def build_layer_alpha_masks(decomposition: ImageLayerDecomposition) -> dict[str, list[list[float]]]:
    layers = {
        "foreground": decomposition.foreground,
        "texture": decomposition.texture,
        "reflections": decomposition.reflections,
        "shadows": decomposition.shadows,
        "specular": decomposition.specular,
        "noise": decomposition.noise,
        "edges": decomposition.edges,
        "ink_mask": decomposition.ink_mask,
        "stroke_mask": decomposition.stroke_mask,
    }
    alphas: dict[str, list[list[float]]] = {}
    for name, layer in layers.items():
        x = np.asarray(layer, dtype=np.float64)
        maxv = np.max(np.abs(x)) if x.size else 0.0
        alphas[name] = (np.abs(x) / max(maxv, 1e-9)).tolist()
    return alphas


def reconstruct_from_layers(
    *,
    background: list[list[float]] | np.ndarray,
    foreground: list[list[float]] | np.ndarray,
    reflections: list[list[float]] | np.ndarray | None = None,
    noise: list[list[float]] | np.ndarray | None = None,
    foreground_sign: float = -1.0,
) -> list[list[float]]:
    bg = np.asarray(background, dtype=np.float64)
    fg = np.asarray(foreground, dtype=np.float64)
    reconstructed = bg + foreground_sign * fg
    if reflections is not None:
        reconstructed = reconstructed + np.asarray(reflections, dtype=np.float64)
    if noise is not None:
        reconstructed = reconstructed + np.asarray(noise, dtype=np.float64)
    return reconstructed.tolist()


def save_layer_image(layer: list[list[float]] | np.ndarray, path: str | Path) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(layer, dtype=np.float64)
    image = Image.fromarray(_normalize_to_uint8(x), mode="L")
    image.save(out_path)
    return str(out_path)


def save_alpha_masks(
    decomposition: ImageLayerDecomposition,
    output_dir: str | Path,
    *,
    prefix: str = "alpha",
    image_format: str = "png",
) -> SavedLayerImages:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    masks = build_layer_alpha_masks(decomposition)
    saved: dict[str, str] = {}
    for name, layer in masks.items():
        saved[name] = save_layer_image(layer, root / f"{prefix}_{name}.{image_format}")
    return SavedLayerImages(output_dir=str(root), paths=saved)


def save_decomposition_layers(
    decomposition: ImageLayerDecomposition,
    output_dir: str | Path,
    *,
    prefix: str = "layer",
    image_format: str = "png",
) -> SavedLayerImages:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    layers = {
        "grayscale": decomposition.grayscale,
        "illumination": decomposition.illumination,
        "background": decomposition.background,
        "foreground": decomposition.foreground,
        "texture": decomposition.texture,
        "reflections": decomposition.reflections,
        "shadows": decomposition.shadows,
        "specular": decomposition.specular,
        "noise": decomposition.noise,
        "edges": decomposition.edges,
        "denoised": decomposition.denoised,
        "ink_mask": decomposition.ink_mask,
        "stroke_mask": decomposition.stroke_mask,
    }
    saved: dict[str, str] = {}
    for name, layer in layers.items():
        saved[name] = save_layer_image(layer, root / f"{prefix}_{name}.{image_format}")
    return SavedLayerImages(output_dir=str(root), paths=saved)


def decompose_image_layers(
    image: list[list[float]] | list[list[list[float]]] | np.ndarray,
    *,
    background_kernel_size: int = 15,
    background_smooth_kernel_size: int = 5,
    illumination_kernel_size: int = 31,
    denoise_kernel_size: int = 3,
    edge_kernel_size: int = 3,
    reflection_percentile: float = 98.0,
    shadow_strength: float = 0.08,
    specular_percentile: float = 99.0,
    foreground_mode: str = "dark",
    segmentation_method: str = "otsu",
    wavelet_family: str | None = None,
    wavelet_level: int = 1,
) -> ImageLayerDecomposition:
    gray = np.asarray(rgb_to_gray(image), dtype=np.float64)
    background = np.asarray(
        estimate_background(
            gray,
            kernel_size=background_kernel_size,
            smooth_kernel_size=background_smooth_kernel_size,
        ),
        dtype=np.float64,
    )
    illumination = np.asarray(estimate_illumination(gray, kernel_size=illumination_kernel_size), dtype=np.float64)
    raw_foreground = np.asarray(extract_foreground(gray, background, mode=foreground_mode), dtype=np.float64)
    reflections = np.asarray(detect_reflections(gray, percentile=reflection_percentile), dtype=np.float64)
    shadows, specular = decompose_shadows_specular(
        gray,
        illumination,
        shadow_strength=shadow_strength,
        specular_percentile=specular_percentile,
    )
    ink_mask, stroke_mask = segment_ink_strokes(gray, background, method=segmentation_method, mode=foreground_mode)
    ink_mask_arr = np.asarray(ink_mask, dtype=np.float64)
    stroke_mask_arr = np.asarray(stroke_mask, dtype=np.float64)
    shadows_arr = np.asarray(shadows, dtype=np.float64)
    specular_arr = np.asarray(specular, dtype=np.float64)
    foreground_clean = raw_foreground * (1.0 - reflections) * np.maximum(ink_mask_arr, 1e-9)
    denoised = np.asarray(denoise_image(gray, kernel_size=denoise_kernel_size), dtype=np.float64)
    noise = np.asarray(estimate_noise(gray, denoised), dtype=np.float64)
    texture = np.asarray(extract_texture(gray, background), dtype=np.float64)
    edges = np.asarray(simple_edges(gray, kernel_size=edge_kernel_size), dtype=np.float64)

    meta: dict[str, Any] = {
        "shape": list(gray.shape),
        "foreground_mode": foreground_mode,
        "segmentation_method": segmentation_method,
        "background_kernel_size": background_kernel_size,
        "background_smooth_kernel_size": background_smooth_kernel_size,
        "illumination_kernel_size": illumination_kernel_size,
        "denoise_kernel_size": denoise_kernel_size,
        "edge_kernel_size": edge_kernel_size,
        "reflection_percentile": reflection_percentile,
        "shadow_strength": shadow_strength,
        "specular_percentile": specular_percentile,
    }
    if wavelet_family is not None:
        subbands = wavelet_subbands_2d(gray, family=wavelet_family, level=wavelet_level)
        meta["wavelet_subbands"] = subbands.to_dict()

    return ImageLayerDecomposition(
        original=np.asarray(image, dtype=np.float64).tolist(),
        grayscale=gray.tolist(),
        illumination=illumination.tolist(),
        background=background.tolist(),
        foreground=foreground_clean.tolist(),
        texture=texture.tolist(),
        reflections=reflections.tolist(),
        shadows=shadows_arr.tolist(),
        specular=specular_arr.tolist(),
        noise=noise.tolist(),
        edges=edges.tolist(),
        denoised=denoised.tolist(),
        ink_mask=ink_mask_arr.tolist(),
        stroke_mask=stroke_mask_arr.tolist(),
        meta=meta,
    )


__all__ = [
    "ImageLayerDecomposition",
    "SavedLayerImages",
    "WaveletSubbands2D",
    "build_layer_alpha_masks",
    "decompose_image_layers",
    "decompose_shadows_specular",
    "denoise_image",
    "detect_reflections",
    "estimate_background",
    "estimate_illumination",
    "estimate_noise",
    "extract_foreground",
    "extract_texture",
    "mean_filter_2d",
    "reconstruct_from_layers",
    "rgb_to_gray",
    "save_alpha_masks",
    "save_decomposition_layers",
    "save_layer_image",
    "segment_ink_strokes",
    "simple_edges",
    "wavelet_subbands_2d",
]
