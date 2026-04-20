"""Utilities to export comparison mosaics for image decomposition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .image_decomposition import ImageLayerDecomposition


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    x = np.asarray(array, dtype=np.float64)
    if x.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    x = x - np.min(x)
    peak = np.max(x)
    if peak <= 1e-12:
        return np.zeros_like(x, dtype=np.uint8)
    return np.clip((255.0 * x / peak).round(), 0, 255).astype(np.uint8)


@dataclass
class ComparisonMosaicResult:
    output_path: str
    tile_names: list[str]
    columns: int
    rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def export_comparison_mosaic(
    decomposition: ImageLayerDecomposition,
    output_path: str | Path,
    *,
    include_layers: tuple[str, ...] = (
        "grayscale",
        "background",
        "foreground",
        "ink_mask",
        "stroke_mask",
        "shadows",
        "specular",
        "edges",
        "denoised",
    ),
    columns: int = 3,
    tile_padding: int = 8,
    title_height: int = 22,
) -> ComparisonMosaicResult:
    if columns <= 0:
        raise ValueError("columns must be positive")
    tiles: list[tuple[str, Image.Image]] = []
    for name in include_layers:
        layer = getattr(decomposition, name)
        arr = _normalize_to_uint8(np.asarray(layer, dtype=np.float64))
        img = Image.fromarray(arr, mode="L").convert("RGB")
        tiles.append((name, img))

    if not tiles:
        raise ValueError("include_layers must not be empty")

    tile_w, tile_h = tiles[0][1].size
    rows = (len(tiles) + columns - 1) // columns
    mosaic_w = columns * tile_w + (columns + 1) * tile_padding
    mosaic_h = rows * (tile_h + title_height) + (rows + 1) * tile_padding
    canvas = Image.new("RGB", (mosaic_w, mosaic_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for idx, (name, tile) in enumerate(tiles):
        row = idx // columns
        col = idx % columns
        x = tile_padding + col * tile_w + col * tile_padding
        y = tile_padding + row * (tile_h + title_height) + row * tile_padding
        canvas.paste(tile, (x, y + title_height))
        draw.text((x, y), name, fill=(230, 230, 230))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return ComparisonMosaicResult(
        output_path=str(out_path),
        tile_names=[name for name, _ in tiles],
        columns=columns,
        rows=rows,
    )


__all__ = ["ComparisonMosaicResult", "export_comparison_mosaic"]
