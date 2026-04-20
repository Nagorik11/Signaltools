"""Separable 2D wavelet packet helpers for images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .utils import ensure_positive_int
from .wavelet_packet import wavelet_filters


@dataclass
class WaveletPacket2DTree:
    nodes: dict[str, list[list[float]]]
    level: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _as_image(image: list[list[float]] | np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("image must be 2D")
    return x



def _row_filter_downsample(x: np.ndarray, filt: np.ndarray) -> np.ndarray:
    tmp = np.stack([np.convolve(row, filt, mode="same")[::2] for row in x], axis=0)
    return tmp



def _col_filter_downsample(x: np.ndarray, filt: np.ndarray) -> np.ndarray:
    tmp = np.stack([np.convolve(col, filt, mode="same")[::2] for col in x.T], axis=0).T
    return tmp



def _row_upsample_filter(x: np.ndarray, filt: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros((x.shape[0], width), dtype=np.float64)
    for i, row in enumerate(x):
        up = np.zeros(width, dtype=np.float64)
        up[::2][: len(row)] = row
        out[i] = np.convolve(up, filt, mode="same")
    return out



def _col_upsample_filter(x: np.ndarray, filt: np.ndarray, height: int) -> np.ndarray:
    out = np.zeros((height, x.shape[1]), dtype=np.float64)
    for j, col in enumerate(x.T):
        up = np.zeros(height, dtype=np.float64)
        up[::2][: len(col)] = col
        out[:, j] = np.convolve(up, filt, mode="same")
    return out



def _analysis_2d(image: np.ndarray, family: str) -> dict[str, np.ndarray]:
    h0, h1, _, _ = wavelet_filters(family)
    low_rows = _row_filter_downsample(image, h0)
    high_rows = _row_filter_downsample(image, h1)
    ll = _col_filter_downsample(low_rows, h0)
    lh = _col_filter_downsample(low_rows, h1)
    hl = _col_filter_downsample(high_rows, h0)
    hh = _col_filter_downsample(high_rows, h1)
    return {"ll": ll, "lh": lh, "hl": hl, "hh": hh}



def _synthesis_2d(ll: np.ndarray, lh: np.ndarray, hl: np.ndarray, hh: np.ndarray, family: str) -> np.ndarray:
    _, _, g0, g1 = wavelet_filters(family)
    height = max(ll.shape[0], lh.shape[0], hl.shape[0], hh.shape[0]) * 2
    width = max(ll.shape[1], lh.shape[1], hl.shape[1], hh.shape[1]) * 2

    low = _col_upsample_filter(ll, g0, height) + _col_upsample_filter(lh, g1, height)
    high = _col_upsample_filter(hl, g0, height) + _col_upsample_filter(hh, g1, height)
    return _row_upsample_filter(low, g0, width) + _row_upsample_filter(high, g1, width)



def wavelet_packet_2d_decompose(image: list[list[float]] | np.ndarray, level: int = 2, family: str = "haar") -> WaveletPacket2DTree:
    """Decompose a 2D image into a full wavelet packet tree."""
    level = ensure_positive_int(level, "level")
    x = _as_image(image)
    nodes: dict[str, list[list[float]]] = {"": x.tolist()}
    shapes: dict[str, tuple[int, int]] = {"": x.shape}
    current = {"": x}
    for _ in range(level):
        nxt: dict[str, np.ndarray] = {}
        for path, img in current.items():
            bands = _analysis_2d(img, family)
            for band_name, band in bands.items():
                child = f"{path}/{band_name}" if path else band_name
                nxt[child] = band
                nodes[child] = band.tolist()
                shapes[child] = band.shape
        current = nxt
    return WaveletPacket2DTree(nodes=nodes, level=level, meta={"wavelet": family, "root_shape": x.shape, "shapes": {k: list(v) for k, v in shapes.items()}})



def wavelet_packet_2d_reconstruct(tree: WaveletPacket2DTree) -> list[list[float]]:
    """Reconstruct an image from a 2D packet tree."""
    family = tree.meta.get("wavelet", "haar")
    current = {k: np.asarray(v, dtype=np.float64) for k, v in tree.nodes.items() if k.count("/") + (1 if k else 0) == tree.level}
    for depth in range(tree.level - 1, -1, -1):
        nxt: dict[str, np.ndarray] = {}
        prefixes = sorted(set(path.rsplit("/", 1)[0] if "/" in path else "" for path in current))
        for prefix in prefixes:
            base = f"{prefix}/" if prefix else ""
            ll = current[base + "ll"]
            lh = current[base + "lh"]
            hl = current[base + "hl"]
            hh = current[base + "hh"]
            nxt[prefix] = _synthesis_2d(ll, lh, hl, hh, family)
        current = nxt
    return current.get("", np.zeros(tree.meta.get("root_shape", (0, 0)), dtype=np.float64)).tolist()


__all__ = ["WaveletPacket2DTree", "wavelet_packet_2d_decompose", "wavelet_packet_2d_reconstruct"]
