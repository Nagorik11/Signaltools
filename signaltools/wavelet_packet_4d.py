"""Separable 4D wavelet packet helpers for spatiotemporal data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .utils import ensure_positive_int
from .wavelet_packet import wavelet_filters


@dataclass
class WaveletPacket4DTree:
    nodes: dict[str, list]
    level: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnisotropicWaveletPacket4DTree:
    nodes: dict[str, list]
    level: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _as_tensor4d(tensor: list | np.ndarray) -> np.ndarray:
    x = np.asarray(tensor, dtype=np.float64)
    if x.ndim != 4:
        raise ValueError("tensor must be 4D")
    return x



def _downsample_axis(x: np.ndarray, filt: np.ndarray, axis: int) -> np.ndarray:
    return np.apply_along_axis(lambda arr: np.convolve(arr, filt, mode="same")[::2], axis, x)



def _upsample_axis(x: np.ndarray, filt: np.ndarray, axis: int, out_len: int) -> np.ndarray:
    def _op(arr: np.ndarray) -> np.ndarray:
        up = np.zeros(out_len, dtype=np.float64)
        up[::2][: len(arr)] = arr
        return np.convolve(up, filt, mode="same")

    return np.apply_along_axis(_op, axis, x)



def _axis_filters(families: tuple[str, str, str, str]) -> dict[int, dict[str, np.ndarray]]:
    filters: dict[int, dict[str, np.ndarray]] = {}
    for axis, family in enumerate(families):
        h0, h1, g0, g1 = wavelet_filters(family)
        filters[axis] = {"l_dec": h0, "h_dec": h1, "l_rec": g0, "h_rec": g1}
    return filters



def _analysis_4d(tensor: np.ndarray, family: str) -> dict[str, np.ndarray]:
    return _analysis_4d_anisotropic(tensor, (family, family, family, family))



def _analysis_4d_anisotropic(tensor: np.ndarray, families: tuple[str, str, str, str]) -> dict[str, np.ndarray]:
    filt = _axis_filters(families)
    bands: dict[str, np.ndarray] = {}
    for a in ["l", "h"]:
        x1 = _downsample_axis(tensor, filt[0][f"{a}_dec"], axis=0)
        for b in ["l", "h"]:
            x2 = _downsample_axis(x1, filt[1][f"{b}_dec"], axis=1)
            for c in ["l", "h"]:
                x3 = _downsample_axis(x2, filt[2][f"{c}_dec"], axis=2)
                for d in ["l", "h"]:
                    x4 = _downsample_axis(x3, filt[3][f"{d}_dec"], axis=3)
                    bands[a + b + c + d] = x4
    return bands



def _synthesis_4d(bands: dict[str, np.ndarray], family: str) -> np.ndarray:
    return _synthesis_4d_anisotropic(bands, (family, family, family, family))



def _synthesis_4d_anisotropic(bands: dict[str, np.ndarray], families: tuple[str, str, str, str]) -> np.ndarray:
    filt = _axis_filters(families)
    ref = next(iter(bands.values()))
    d0, d1, d2, d3 = ref.shape[0] * 2, ref.shape[1] * 2, ref.shape[2] * 2, ref.shape[3] * 2
    accum = np.zeros((d0, d1, d2, d3), dtype=np.float64)
    for key, band in bands.items():
        x = _upsample_axis(band, filt[3][f"{key[3]}_rec"], axis=3, out_len=d3)
        x = _upsample_axis(x, filt[2][f"{key[2]}_rec"], axis=2, out_len=d2)
        x = _upsample_axis(x, filt[1][f"{key[1]}_rec"], axis=1, out_len=d1)
        x = _upsample_axis(x, filt[0][f"{key[0]}_rec"], axis=0, out_len=d0)
        accum += x
    return accum



def wavelet_packet_4d_decompose(tensor: list | np.ndarray, level: int = 1, family: str = "haar") -> WaveletPacket4DTree:
    """Decompose a 4D spatiotemporal tensor into a full packet tree."""
    level = ensure_positive_int(level, "level")
    x = _as_tensor4d(tensor)
    nodes: dict[str, list] = {"": x.tolist()}
    shapes: dict[str, tuple[int, int, int, int]] = {"": x.shape}
    current = {"": x}
    for _ in range(level):
        nxt: dict[str, np.ndarray] = {}
        for path, ten in current.items():
            bands = _analysis_4d(ten, family)
            for band_name, band in bands.items():
                child = f"{path}/{band_name}" if path else band_name
                nxt[child] = band
                nodes[child] = band.tolist()
                shapes[child] = band.shape
        current = nxt
    return WaveletPacket4DTree(nodes=nodes, level=level, meta={"wavelet": family, "root_shape": x.shape, "shapes": {k: list(v) for k, v in shapes.items()}})



def anisotropic_wavelet_packet_4d_decompose(tensor: list | np.ndarray, level: int = 1, families: tuple[str, str, str, str] = ("haar", "haar", "haar", "haar")) -> AnisotropicWaveletPacket4DTree:
    """Decompose a 4D tensor with per-axis wavelet families."""
    level = ensure_positive_int(level, "level")
    x = _as_tensor4d(tensor)
    if len(families) != 4:
        raise ValueError("families must contain exactly 4 wavelet names")
    nodes: dict[str, list] = {"": x.tolist()}
    shapes: dict[str, tuple[int, int, int, int]] = {"": x.shape}
    current = {"": x}
    for _ in range(level):
        nxt: dict[str, np.ndarray] = {}
        for path, ten in current.items():
            bands = _analysis_4d_anisotropic(ten, families)
            for band_name, band in bands.items():
                child = f"{path}/{band_name}" if path else band_name
                nxt[child] = band
                nodes[child] = band.tolist()
                shapes[child] = band.shape
        current = nxt
    return AnisotropicWaveletPacket4DTree(nodes=nodes, level=level, meta={"families": list(families), "root_shape": x.shape, "shapes": {k: list(v) for k, v in shapes.items()}})



def wavelet_packet_4d_reconstruct(tree: WaveletPacket4DTree) -> list:
    """Reconstruct a 4D tensor from a packet tree."""
    family = tree.meta.get("wavelet", "haar")
    current = {k: np.asarray(v, dtype=np.float64) for k, v in tree.nodes.items() if k.count("/") + (1 if k else 0) == tree.level}
    band_names = [
        "llll", "lllh", "llhl", "llhh", "lhll", "lhlh", "lhhl", "lhhh",
        "hlll", "hllh", "hlhl", "hlhh", "hhll", "hhlh", "hhhl", "hhhh",
    ]
    for _depth in range(tree.level - 1, -1, -1):
        nxt: dict[str, np.ndarray] = {}
        prefixes = sorted(set(path.rsplit("/", 1)[0] if "/" in path else "" for path in current))
        for prefix in prefixes:
            base = f"{prefix}/" if prefix else ""
            bands = {name: current[base + name] for name in band_names}
            nxt[prefix] = _synthesis_4d(bands, family)
        current = nxt
    root_shape = tree.meta.get("root_shape", (0, 0, 0, 0))
    return current.get("", np.zeros(root_shape, dtype=np.float64)).tolist()



def anisotropic_wavelet_packet_4d_reconstruct(tree: AnisotropicWaveletPacket4DTree) -> list:
    """Reconstruct a 4D tensor from an anisotropic packet tree."""
    families = tuple(tree.meta.get("families", ["haar", "haar", "haar", "haar"]))
    current = {k: np.asarray(v, dtype=np.float64) for k, v in tree.nodes.items() if k.count("/") + (1 if k else 0) == tree.level}
    band_names = [
        "llll", "lllh", "llhl", "llhh", "lhll", "lhlh", "lhhl", "lhhh",
        "hlll", "hllh", "hlhl", "hlhh", "hhll", "hhlh", "hhhl", "hhhh",
    ]
    for _depth in range(tree.level - 1, -1, -1):
        nxt: dict[str, np.ndarray] = {}
        prefixes = sorted(set(path.rsplit("/", 1)[0] if "/" in path else "" for path in current))
        for prefix in prefixes:
            base = f"{prefix}/" if prefix else ""
            bands = {name: current[base + name] for name in band_names}
            nxt[prefix] = _synthesis_4d_anisotropic(bands, families)
        current = nxt
    root_shape = tree.meta.get("root_shape", (0, 0, 0, 0))
    return current.get("", np.zeros(root_shape, dtype=np.float64)).tolist()


__all__ = [
    "WaveletPacket4DTree",
    "AnisotropicWaveletPacket4DTree",
    "wavelet_packet_4d_decompose",
    "wavelet_packet_4d_reconstruct",
    "anisotropic_wavelet_packet_4d_decompose",
    "anisotropic_wavelet_packet_4d_reconstruct",
]
