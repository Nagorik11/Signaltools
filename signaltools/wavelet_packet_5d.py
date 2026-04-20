"""Separable 5D wavelet packet helpers for multimodal and hyperspectral tensors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .utils import ensure_positive_int
from .wavelet_packet import available_wavelet_families, wavelet_family_kind, wavelet_filters


@dataclass
class WaveletPacket5DTree:
    nodes: dict[str, list]
    level: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnisotropicWaveletPacket5DTree:
    nodes: dict[str, list]
    level: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveWaveletPacket5DTree:
    nodes: dict[str, list]
    level: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BlockAdaptiveWaveletPacket5DTree:
    nodes: dict[str, list]
    level: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SpatiallyAdaptiveWaveletPacket5DTree:
    nodes: dict[str, list]
    level: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SubbandAdaptiveWaveletPacket5DTree:
    nodes: dict[str, list]
    level: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegularizedAdaptiveWaveletPacket5DTree:
    nodes: dict[str, list]
    level: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SubbandAttentiveWaveletPacket5DTree:
    nodes: dict[str, list]
    level: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CrossBranchAttentiveWaveletPacket5DTree:
    nodes: dict[str, list]
    level: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WeightedMultiObjectiveWaveletPacket5DTree:
    nodes: dict[str, list]
    level: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LearnableWeightedMultiObjectiveWaveletPacket5DTree:
    nodes: dict[str, list]
    level: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LevelAttentiveWaveletPacket5DTree:
    nodes: dict[str, list]
    level: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _as_tensor5d(tensor: list | np.ndarray) -> np.ndarray:
    x = np.asarray(tensor, dtype=np.float64)
    if x.ndim != 5:
        raise ValueError("tensor must be 5D")
    return x



def _downsample_axis(x: np.ndarray, filt: np.ndarray, axis: int) -> np.ndarray:
    return np.apply_along_axis(lambda arr: np.convolve(arr, filt, mode="same")[::2], axis, x)



def _upsample_axis(x: np.ndarray, filt: np.ndarray, axis: int, out_len: int) -> np.ndarray:
    def _op(arr: np.ndarray) -> np.ndarray:
        up = np.zeros(out_len, dtype=np.float64)
        up[::2][: len(arr)] = arr
        return np.convolve(up, filt, mode="same")

    return np.apply_along_axis(_op, axis, x)



def _axis_filters(families: tuple[str, str, str, str, str]) -> dict[int, dict[str, np.ndarray]]:
    filters: dict[int, dict[str, np.ndarray]] = {}
    for axis, family in enumerate(families):
        h0, h1, g0, g1 = wavelet_filters(family)
        filters[axis] = {"l_dec": h0, "h_dec": h1, "l_rec": g0, "h_rec": g1}
    return filters



def _analysis_5d(tensor: np.ndarray, family: str) -> dict[str, np.ndarray]:
    return _analysis_5d_anisotropic(tensor, (family, family, family, family, family))



def _analysis_5d_anisotropic(tensor: np.ndarray, families: tuple[str, str, str, str, str]) -> dict[str, np.ndarray]:
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
                    for e in ["l", "h"]:
                        x5 = _downsample_axis(x4, filt[4][f"{e}_dec"], axis=4)
                        bands[a + b + c + d + e] = x5
    return bands



def _synthesis_5d(bands: dict[str, np.ndarray], family: str) -> np.ndarray:
    return _synthesis_5d_anisotropic(bands, (family, family, family, family, family))



def _synthesis_5d_anisotropic(bands: dict[str, np.ndarray], families: tuple[str, str, str, str, str]) -> np.ndarray:
    filt = _axis_filters(families)
    ref = next(iter(bands.values()))
    d0, d1, d2, d3, d4 = ref.shape[0] * 2, ref.shape[1] * 2, ref.shape[2] * 2, ref.shape[3] * 2, ref.shape[4] * 2
    accum = np.zeros((d0, d1, d2, d3, d4), dtype=np.float64)
    for key, band in bands.items():
        x = _upsample_axis(band, filt[4][f"{key[4]}_rec"], axis=4, out_len=d4)
        x = _upsample_axis(x, filt[3][f"{key[3]}_rec"], axis=3, out_len=d3)
        x = _upsample_axis(x, filt[2][f"{key[2]}_rec"], axis=2, out_len=d2)
        x = _upsample_axis(x, filt[1][f"{key[1]}_rec"], axis=1, out_len=d1)
        x = _upsample_axis(x, filt[0][f"{key[0]}_rec"], axis=0, out_len=d0)
        accum += x
    return accum



def wavelet_packet_5d_decompose(tensor: list | np.ndarray, level: int = 1, family: str = "haar") -> WaveletPacket5DTree:
    """Decompose a 5D tensor into a full packet tree."""
    level = ensure_positive_int(level, "level")
    x = _as_tensor5d(tensor)
    nodes: dict[str, list] = {"": x.tolist()}
    shapes: dict[str, tuple[int, int, int, int, int]] = {"": x.shape}
    current = {"": x}
    for _ in range(level):
        nxt: dict[str, np.ndarray] = {}
        for path, ten in current.items():
            bands = _analysis_5d(ten, family)
            for band_name, band in bands.items():
                child = f"{path}/{band_name}" if path else band_name
                nxt[child] = band
                nodes[child] = band.tolist()
                shapes[child] = band.shape
        current = nxt
    return WaveletPacket5DTree(nodes=nodes, level=level, meta={"wavelet": family, "kind": wavelet_family_kind(family), "supported": available_wavelet_families(), "root_shape": x.shape, "shapes": {k: list(v) for k, v in shapes.items()}})



def anisotropic_wavelet_packet_5d_decompose(tensor: list | np.ndarray, level: int = 1, families: tuple[str, str, str, str, str] = ("haar", "haar", "haar", "haar", "haar")) -> AnisotropicWaveletPacket5DTree:
    """Decompose a 5D tensor with per-axis wavelet families."""
    level = ensure_positive_int(level, "level")
    x = _as_tensor5d(tensor)
    if len(families) != 5:
        raise ValueError("families must contain exactly 5 wavelet names")
    nodes: dict[str, list] = {"": x.tolist()}
    shapes: dict[str, tuple[int, int, int, int, int]] = {"": x.shape}
    current = {"": x}
    for _ in range(level):
        nxt: dict[str, np.ndarray] = {}
        for path, ten in current.items():
            bands = _analysis_5d_anisotropic(ten, families)
            for band_name, band in bands.items():
                child = f"{path}/{band_name}" if path else band_name
                nxt[child] = band
                nodes[child] = band.tolist()
                shapes[child] = band.shape
        current = nxt
    return AnisotropicWaveletPacket5DTree(nodes=nodes, level=level, meta={"families": list(families), "family_kinds": [wavelet_family_kind(f) for f in families], "supported": available_wavelet_families(), "root_shape": x.shape, "shapes": {k: list(v) for k, v in shapes.items()}})



def wavelet_packet_5d_reconstruct(tree: WaveletPacket5DTree) -> list:
    """Reconstruct a 5D tensor from a packet tree."""
    family = tree.meta.get("wavelet", "haar")
    current = {k: np.asarray(v, dtype=np.float64) for k, v in tree.nodes.items() if k.count("/") + (1 if k else 0) == tree.level}
    band_names = [format(i, "05b").replace("0", "l").replace("1", "h") for i in range(32)]
    for _depth in range(tree.level - 1, -1, -1):
        nxt: dict[str, np.ndarray] = {}
        prefixes = sorted(set(path.rsplit("/", 1)[0] if "/" in path else "" for path in current))
        for prefix in prefixes:
            base = f"{prefix}/" if prefix else ""
            bands = {name: current[base + name] for name in band_names}
            nxt[prefix] = _synthesis_5d(bands, family)
        current = nxt
    root_shape = tree.meta.get("root_shape", (0, 0, 0, 0, 0))
    return current.get("", np.zeros(root_shape, dtype=np.float64)).tolist()



def anisotropic_wavelet_packet_5d_reconstruct(tree: AnisotropicWaveletPacket5DTree) -> list:
    """Reconstruct a 5D tensor from an anisotropic packet tree."""
    families = tuple(tree.meta.get("families", ["haar", "haar", "haar", "haar", "haar"]))
    current = {k: np.asarray(v, dtype=np.float64) for k, v in tree.nodes.items() if k.count("/") + (1 if k else 0) == tree.level}
    band_names = [format(i, "05b").replace("0", "l").replace("1", "h") for i in range(32)]
    for _depth in range(tree.level - 1, -1, -1):
        nxt: dict[str, np.ndarray] = {}
        prefixes = sorted(set(path.rsplit("/", 1)[0] if "/" in path else "" for path in current))
        for prefix in prefixes:
            base = f"{prefix}/" if prefix else ""
            bands = {name: current[base + name] for name in band_names}
            nxt[prefix] = _synthesis_5d_anisotropic(bands, families)
        current = nxt
    root_shape = tree.meta.get("root_shape", (0, 0, 0, 0, 0))
    return current.get("", np.zeros(root_shape, dtype=np.float64)).tolist()



def select_wavelet_family_per_axis_5d(
    tensor: list | np.ndarray,
    candidate_families: list[str] | tuple[str, ...] | None = None,
    criterion: str = "low_high_ratio",
) -> dict[str, Any]:
    """Select a wavelet family independently for each of the five axes."""
    x = _as_tensor5d(tensor)
    families = list(candidate_families or available_wavelet_families())
    scores: list[dict[str, float]] = []
    selected: list[str] = []
    for axis in range(5):
        moved = np.moveaxis(x, axis, 0)
        profile = np.mean(moved, axis=tuple(range(1, moved.ndim)))
        axis_scores: dict[str, float] = {}
        for family in families:
            h0, h1, _, _ = wavelet_filters(family)
            low = np.convolve(profile, h0, mode="same")[::2]
            high = np.convolve(profile, h1, mode="same")[::2]
            low_e = float(np.sum(np.square(low)))
            high_e = float(np.sum(np.square(high)))
            if criterion == "low_high_ratio":
                score = low_e / (high_e + 1e-12)
            elif criterion == "energy_gap":
                score = abs(low_e - high_e)
            else:
                raise ValueError("criterion must be 'low_high_ratio' or 'energy_gap'")
            axis_scores[family] = score
        best = max(axis_scores, key=axis_scores.get)
        selected.append(best)
        scores.append(axis_scores)
    return {
        "families": selected,
        "family_kinds": [wavelet_family_kind(f) for f in selected],
        "scores": scores,
        "criterion": criterion,
        "supported": available_wavelet_families(),
    }



def adaptive_wavelet_packet_5d_decompose(
    tensor: list | np.ndarray,
    level: int = 1,
    candidate_families: list[str] | tuple[str, ...] | None = None,
    criterion: str = "low_high_ratio",
) -> AdaptiveWaveletPacket5DTree:
    """Adaptively choose one family per axis, then perform anisotropic 5D packet decomposition."""
    selection = select_wavelet_family_per_axis_5d(tensor, candidate_families=candidate_families, criterion=criterion)
    tree = anisotropic_wavelet_packet_5d_decompose(tensor, level=level, families=tuple(selection["families"]))
    meta = dict(tree.meta)
    meta.update({
        "selection_scores": selection["scores"],
        "criterion": selection["criterion"],
        "adaptive": True,
    })
    return AdaptiveWaveletPacket5DTree(nodes=tree.nodes, level=tree.level, meta=meta)



def adaptive_wavelet_packet_5d_reconstruct(tree: AdaptiveWaveletPacket5DTree) -> list:
    """Reconstruct a 5D tensor from an adaptively selected anisotropic packet tree."""
    families = tuple(tree.meta.get("families", ["haar", "haar", "haar", "haar", "haar"]))
    proxy = AnisotropicWaveletPacket5DTree(nodes=tree.nodes, level=tree.level, meta={"families": list(families), "root_shape": tree.meta.get("root_shape", (0, 0, 0, 0, 0))})
    return anisotropic_wavelet_packet_5d_reconstruct(proxy)



def _iter_blocks_5d(x: np.ndarray, block_shape: tuple[int, int, int, int, int]):
    for i0 in range(0, x.shape[0], block_shape[0]):
        for i1 in range(0, x.shape[1], block_shape[1]):
            for i2 in range(0, x.shape[2], block_shape[2]):
                for i3 in range(0, x.shape[3], block_shape[3]):
                    for i4 in range(0, x.shape[4], block_shape[4]):
                        yield (i0, i1, i2, i3, i4), x[
                            i0:i0+block_shape[0],
                            i1:i1+block_shape[1],
                            i2:i2+block_shape[2],
                            i3:i3+block_shape[3],
                            i4:i4+block_shape[4],
                        ]



def select_wavelet_family_per_axis_5d_local_blocks(
    tensor: list | np.ndarray,
    candidate_families: list[str] | tuple[str, ...] | None = None,
    block_shape: tuple[int, int, int, int, int] = (1, 1, 4, 4, 4),
    criterion: str = "local_variation",
) -> dict[str, Any]:
    """Select one family per axis by aggregating local block metrics."""
    x = _as_tensor5d(tensor)
    families = list(candidate_families or available_wavelet_families())
    for size in block_shape:
        ensure_positive_int(size, "block_shape_item")
    axis_totals = [dict.fromkeys(families, 0.0) for _ in range(5)]
    block_reports = []
    for block_index, block in _iter_blocks_5d(x, block_shape):
        report = select_wavelet_family_per_axis_5d(block, candidate_families=families, criterion="low_high_ratio")
        local_strength = float(np.mean(np.abs(np.diff(block.reshape(-1))))) if block.size > 1 else 0.0
        if criterion not in {"local_variation", "low_high_ratio"}:
            raise ValueError("criterion must be 'local_variation' or 'low_high_ratio'")
        for axis in range(5):
            for family, score in report["scores"][axis].items():
                weighted = float(score) * (1.0 + local_strength if criterion == "local_variation" else 1.0)
                axis_totals[axis][family] += weighted
        block_reports.append({"index": list(block_index), "families": report["families"], "local_strength": local_strength})
    selected = [max(scores, key=scores.get) for scores in axis_totals]
    return {
        "families": selected,
        "family_kinds": [wavelet_family_kind(f) for f in selected],
        "scores": axis_totals,
        "criterion": criterion,
        "block_shape": list(block_shape),
        "blocks": block_reports,
        "supported": available_wavelet_families(),
    }



def adaptive_blockwise_wavelet_packet_5d_decompose(
    tensor: list | np.ndarray,
    level: int = 1,
    candidate_families: list[str] | tuple[str, ...] | None = None,
    block_shape: tuple[int, int, int, int, int] = (1, 1, 4, 4, 4),
    criterion: str = "local_variation",
) -> BlockAdaptiveWaveletPacket5DTree:
    """Adaptively choose wavelet families per axis from local block metrics."""
    selection = select_wavelet_family_per_axis_5d_local_blocks(
        tensor,
        candidate_families=candidate_families,
        block_shape=block_shape,
        criterion=criterion,
    )
    tree = anisotropic_wavelet_packet_5d_decompose(tensor, level=level, families=tuple(selection["families"]))
    meta = dict(tree.meta)
    meta.update({
        "selection_scores": selection["scores"],
        "criterion": selection["criterion"],
        "block_shape": selection["block_shape"],
        "blocks": selection["blocks"],
        "adaptive_blockwise": True,
    })
    return BlockAdaptiveWaveletPacket5DTree(nodes=tree.nodes, level=tree.level, meta=meta)



def adaptive_blockwise_wavelet_packet_5d_reconstruct(tree: BlockAdaptiveWaveletPacket5DTree) -> list:
    """Reconstruct a 5D tensor from a locally guided adaptive packet tree."""
    families = tuple(tree.meta.get("families", ["haar", "haar", "haar", "haar", "haar"]))
    proxy = AnisotropicWaveletPacket5DTree(nodes=tree.nodes, level=tree.level, meta={"families": list(families), "root_shape": tree.meta.get("root_shape", (0, 0, 0, 0, 0))})
    return anisotropic_wavelet_packet_5d_reconstruct(proxy)



def spatially_variable_wavelet_packet_5d_decompose(
    tensor: list | np.ndarray,
    level: int = 1,
    candidate_families: list[str] | tuple[str, ...] | None = None,
    block_shape: tuple[int, int, int, int, int] = (1, 1, 2, 2, 2),
    criterion: str = "local_variation",
) -> SpatiallyAdaptiveWaveletPacket5DTree:
    """Decompose a 5D tensor while selecting families independently at each tree node."""
    level = ensure_positive_int(level, "level")
    x = _as_tensor5d(tensor)
    nodes: dict[str, list] = {"": x.tolist()}
    current = {"": x}
    node_families: dict[str, list[str]] = {}
    node_scores: dict[str, Any] = {}
    node_shapes: dict[str, list[int]] = {"": list(x.shape)}
    for _depth in range(level):
        nxt: dict[str, np.ndarray] = {}
        for path, ten in current.items():
            selection = select_wavelet_family_per_axis_5d_local_blocks(
                ten,
                candidate_families=candidate_families,
                block_shape=block_shape,
                criterion=criterion,
            )
            families = tuple(selection["families"])
            node_families[path] = list(families)
            node_scores[path] = selection["scores"]
            bands = _analysis_5d_anisotropic(ten, families)
            for band_name, band in bands.items():
                child = f"{path}/{band_name}" if path else band_name
                nxt[child] = band
                nodes[child] = band.tolist()
                node_shapes[child] = list(band.shape)
        current = nxt
    return SpatiallyAdaptiveWaveletPacket5DTree(
        nodes=nodes,
        level=level,
        meta={
            "root_shape": list(x.shape),
            "node_families": node_families,
            "node_scores": node_scores,
            "node_shapes": node_shapes,
            "block_shape": list(block_shape),
            "criterion": criterion,
            "adaptive_spatial": True,
            "supported": available_wavelet_families(),
        },
    )



def spatially_variable_wavelet_packet_5d_reconstruct(tree: SpatiallyAdaptiveWaveletPacket5DTree) -> list:
    """Reconstruct a 5D tensor from a node-wise adaptive packet tree."""
    current = {k: np.asarray(v, dtype=np.float64) for k, v in tree.nodes.items() if k.count("/") + (1 if k else 0) == tree.level}
    band_names = [format(i, "05b").replace("0", "l").replace("1", "h") for i in range(32)]
    node_families = tree.meta.get("node_families", {})
    for _depth in range(tree.level - 1, -1, -1):
        nxt: dict[str, np.ndarray] = {}
        prefixes = sorted(set(path.rsplit("/", 1)[0] if "/" in path else "" for path in current))
        for prefix in prefixes:
            base = f"{prefix}/" if prefix else ""
            bands = {name: current[base + name] for name in band_names}
            families = tuple(node_families.get(prefix, ["haar", "haar", "haar", "haar", "haar"]))
            nxt[prefix] = _synthesis_5d_anisotropic(bands, families)
        current = nxt
    root_shape = tuple(tree.meta.get("root_shape", (0, 0, 0, 0, 0)))
    return current.get("", np.zeros(root_shape, dtype=np.float64)).tolist()



def subband_adaptive_wavelet_packet_5d_decompose(
    tensor: list | np.ndarray,
    level: int = 1,
    candidate_families: list[str] | tuple[str, ...] | None = None,
    block_shape: tuple[int, int, int, int, int] = (1, 1, 2, 2, 2),
    criterion: str = "local_variation",
    seed_family: str = "haar",
) -> SubbandAdaptiveWaveletPacket5DTree:
    """Adapt families separately for each subband child throughout the packet tree."""
    level = ensure_positive_int(level, "level")
    x = _as_tensor5d(tensor)
    nodes: dict[str, list] = {"": x.tolist()}
    current = {"": x}
    node_families: dict[str, list[str]] = {"": [seed_family] * 5}
    node_scores: dict[str, Any] = {}
    child_presets: dict[str, list[str]] = {}
    for _depth in range(level):
        nxt: dict[str, np.ndarray] = {}
        next_presets: dict[str, list[str]] = {}
        for path, ten in current.items():
            families = tuple(child_presets.get(path, node_families.get(path, [seed_family] * 5)))
            bands = _analysis_5d_anisotropic(ten, families)
            for band_name, band in bands.items():
                child = f"{path}/{band_name}" if path else band_name
                nxt[child] = band
                nodes[child] = band.tolist()
                selection = select_wavelet_family_per_axis_5d_local_blocks(
                    band,
                    candidate_families=candidate_families,
                    block_shape=block_shape,
                    criterion=criterion,
                )
                next_presets[child] = selection["families"]
                node_families[child] = selection["families"]
                node_scores[child] = {"band": band_name, "scores": selection["scores"]}
        current = nxt
        child_presets = next_presets
    return SubbandAdaptiveWaveletPacket5DTree(
        nodes=nodes,
        level=level,
        meta={
            "root_shape": list(x.shape),
            "node_families": node_families,
            "node_scores": node_scores,
            "block_shape": list(block_shape),
            "criterion": criterion,
            "seed_family": seed_family,
            "adaptive_subband": True,
            "supported": available_wavelet_families(),
        },
    )



def subband_adaptive_wavelet_packet_5d_reconstruct(tree: SubbandAdaptiveWaveletPacket5DTree) -> list:
    """Reconstruct a 5D tensor from a subband-adaptive packet tree."""
    current = {k: np.asarray(v, dtype=np.float64) for k, v in tree.nodes.items() if k.count("/") + (1 if k else 0) == tree.level}
    band_names = [format(i, "05b").replace("0", "l").replace("1", "h") for i in range(32)]
    node_families = tree.meta.get("node_families", {})
    for _depth in range(tree.level - 1, -1, -1):
        nxt: dict[str, np.ndarray] = {}
        prefixes = sorted(set(path.rsplit("/", 1)[0] if "/" in path else "" for path in current))
        for prefix in prefixes:
            base = f"{prefix}/" if prefix else ""
            bands = {name: current[base + name] for name in band_names}
            parent_families = node_families.get(prefix, [tree.meta.get("seed_family", "haar")] * 5)
            parent_families = tuple(parent_families)
            nxt[prefix] = _synthesis_5d_anisotropic(bands, parent_families)
        current = nxt
    root_shape = tuple(tree.meta.get("root_shape", (0, 0, 0, 0, 0)))
    return current.get("", np.zeros(root_shape, dtype=np.float64)).tolist()



def _wavelet_complexity_cost(family: str) -> float:
    h0, _, g0, _ = wavelet_filters(family)
    return float(len(h0) + len(g0))



def regularized_select_wavelet_family_per_axis_5d(
    tensor: list | np.ndarray,
    candidate_families: list[str] | tuple[str, ...] | None = None,
    block_shape: tuple[int, int, int, int, int] = (1, 1, 4, 4, 4),
    criterion: str = "local_variation",
    complexity_lambda: float = 0.05,
) -> dict[str, Any]:
    """Select families per axis with a local-score minus complexity penalty objective."""
    report = select_wavelet_family_per_axis_5d_local_blocks(
        tensor,
        candidate_families=candidate_families,
        block_shape=block_shape,
        criterion=criterion,
    )
    regularized_scores=[]
    selected=[]
    for axis_scores in report["scores"]:
        new_scores={family: float(score) - complexity_lambda * _wavelet_complexity_cost(family) for family, score in axis_scores.items()}
        regularized_scores.append(new_scores)
        selected.append(max(new_scores, key=new_scores.get))
    return {
        "families": selected,
        "family_kinds": [wavelet_family_kind(f) for f in selected],
        "scores": regularized_scores,
        "criterion": criterion,
        "complexity_lambda": complexity_lambda,
        "block_shape": list(block_shape),
        "supported": available_wavelet_families(),
    }



def regularized_adaptive_wavelet_packet_5d_decompose(
    tensor: list | np.ndarray,
    level: int = 1,
    candidate_families: list[str] | tuple[str, ...] | None = None,
    block_shape: tuple[int, int, int, int, int] = (1, 1, 4, 4, 4),
    criterion: str = "local_variation",
    complexity_lambda: float = 0.05,
) -> RegularizedAdaptiveWaveletPacket5DTree:
    """Adaptive 5D packet decomposition with complexity regularization."""
    selection = regularized_select_wavelet_family_per_axis_5d(
        tensor,
        candidate_families=candidate_families,
        block_shape=block_shape,
        criterion=criterion,
        complexity_lambda=complexity_lambda,
    )
    tree = anisotropic_wavelet_packet_5d_decompose(tensor, level=level, families=tuple(selection["families"]))
    meta = dict(tree.meta)
    meta.update({
        "selection_scores": selection["scores"],
        "criterion": criterion,
        "complexity_lambda": complexity_lambda,
        "adaptive_regularized": True,
    })
    return RegularizedAdaptiveWaveletPacket5DTree(nodes=tree.nodes, level=tree.level, meta=meta)



def regularized_adaptive_wavelet_packet_5d_reconstruct(tree: RegularizedAdaptiveWaveletPacket5DTree) -> list:
    """Reconstruct a regularized adaptive 5D packet tree."""
    proxy = AnisotropicWaveletPacket5DTree(nodes=tree.nodes, level=tree.level, meta={"families": tree.meta.get("families", ["haar"] * 5), "root_shape": tree.meta.get("root_shape", (0, 0, 0, 0, 0))})
    return anisotropic_wavelet_packet_5d_reconstruct(proxy)



def subband_attentive_wavelet_packet_5d_decompose(
    tensor: list | np.ndarray,
    level: int = 1,
    candidate_families: list[str] | tuple[str, ...] | None = None,
    block_shape: tuple[int, int, int, int, int] = (1, 1, 2, 2, 2),
    criterion: str = "local_variation",
    seed_family: str = "haar",
) -> SubbandAttentiveWaveletPacket5DTree:
    """Decompose a 5D tensor while mixing sibling subbands through attention."""
    level = ensure_positive_int(level, "level")
    x = _as_tensor5d(tensor)
    nodes={"": x.tolist()}
    current={"": x}
    node_families={"": [seed_family]*5}
    subband_attention={}
    child_presets={}
    band_names = [format(i, "05b").replace("0", "l").replace("1", "h") for i in range(32)]
    for _depth in range(level):
        nxt={}
        next_presets={}
        for path, ten in current.items():
            families = tuple(child_presets.get(path, node_families.get(path, [seed_family]*5)))
            raw_bands = _analysis_5d_anisotropic(ten, families)
            reps = np.stack([np.asarray(raw_bands[name], dtype=np.float64).reshape(-1) for name in band_names], axis=0)
            norms = np.linalg.norm(reps, axis=1, keepdims=True)
            sims = reps @ reps.T / np.maximum(norms @ norms.T, 1e-12)
            attn = np.exp(sims - np.max(sims, axis=1, keepdims=True))
            attn = attn / np.maximum(np.sum(attn, axis=1, keepdims=True), 1e-12)
            subband_attention[path] = attn.tolist()
            mixed_bands = {}
            stacked = np.stack([raw_bands[name] for name in band_names], axis=0)
            mixed = np.einsum('ab,bijklm->aijklm', attn, stacked)
            for idx, band_name in enumerate(band_names):
                mixed_bands[band_name] = mixed[idx]
                child = f"{path}/{band_name}" if path else band_name
                nxt[child] = mixed[idx]
                nodes[child] = mixed[idx].tolist()
                selection = select_wavelet_family_per_axis_5d_local_blocks(mixed[idx], candidate_families=candidate_families, block_shape=block_shape, criterion=criterion)
                next_presets[child] = selection["families"]
                node_families[child] = selection["families"]
        current = nxt
        child_presets = next_presets
    return SubbandAttentiveWaveletPacket5DTree(nodes=nodes, level=level, meta={"root_shape": list(x.shape), "node_families": node_families, "subband_attention": subband_attention, "block_shape": list(block_shape), "criterion": criterion, "seed_family": seed_family, "attentive_subbands": True, "supported": available_wavelet_families()})



def subband_attentive_wavelet_packet_5d_reconstruct(tree: SubbandAttentiveWaveletPacket5DTree) -> list:
    """Reconstruct a subband-attentive 5D packet tree."""
    proxy = SubbandAdaptiveWaveletPacket5DTree(nodes=tree.nodes, level=tree.level, meta={"root_shape": tree.meta.get("root_shape", (0,0,0,0,0)), "node_families": tree.meta.get("node_families", {}), "seed_family": tree.meta.get("seed_family", "haar")})
    return subband_adaptive_wavelet_packet_5d_reconstruct(proxy)



def _wavelet_stability_cost(family: str) -> float:
    h0, _, g0, _ = wavelet_filters(family)
    return float(np.sum(np.abs(np.diff(h0))) + np.sum(np.abs(np.diff(g0))))



def weighted_multiobjective_select_wavelet_family_per_axis_5d(
    tensor: list | np.ndarray,
    candidate_families: list[str] | tuple[str, ...] | None = None,
    block_shape: tuple[int, int, int, int, int] = (1, 1, 4, 4, 4),
    criterion: str = "local_variation",
    axis_weights: list[float] | tuple[float, ...] | None = None,
    precision_weight: float = 1.0,
    cost_weight: float = 0.05,
    stability_weight: float = 0.05,
) -> dict[str, Any]:
    """Select families per axis with configurable weights for precision, cost and stability."""
    report = select_wavelet_family_per_axis_5d_local_blocks(
        tensor,
        candidate_families=candidate_families,
        block_shape=block_shape,
        criterion=criterion,
    )
    weights = list(axis_weights or [1.0] * 5)
    if len(weights) != 5:
        raise ValueError("axis_weights must have length 5")
    scores_out=[]
    selected=[]
    for axis, axis_scores in enumerate(report["scores"]):
        axis_map={}
        for family, fit in axis_scores.items():
            score = weights[axis] * (precision_weight * float(fit)) - cost_weight * _wavelet_complexity_cost(family) - stability_weight * _wavelet_stability_cost(family)
            axis_map[family] = score
        scores_out.append(axis_map)
        selected.append(max(axis_map, key=axis_map.get))
    return {
        "families": selected,
        "family_kinds": [wavelet_family_kind(f) for f in selected],
        "scores": scores_out,
        "criterion": criterion,
        "axis_weights": weights,
        "precision_weight": precision_weight,
        "cost_weight": cost_weight,
        "stability_weight": stability_weight,
        "block_shape": list(block_shape),
        "supported": available_wavelet_families(),
    }



def weighted_multiobjective_wavelet_packet_5d_decompose(
    tensor: list | np.ndarray,
    level: int = 1,
    candidate_families: list[str] | tuple[str, ...] | None = None,
    block_shape: tuple[int, int, int, int, int] = (1, 1, 4, 4, 4),
    criterion: str = "local_variation",
    axis_weights: list[float] | tuple[float, ...] | None = None,
    precision_weight: float = 1.0,
    cost_weight: float = 0.05,
    stability_weight: float = 0.05,
) -> WeightedMultiObjectiveWaveletPacket5DTree:
    sel = weighted_multiobjective_select_wavelet_family_per_axis_5d(
        tensor,
        candidate_families=candidate_families,
        block_shape=block_shape,
        criterion=criterion,
        axis_weights=axis_weights,
        precision_weight=precision_weight,
        cost_weight=cost_weight,
        stability_weight=stability_weight,
    )
    tree = anisotropic_wavelet_packet_5d_decompose(tensor, level=level, families=tuple(sel["families"]))
    meta = dict(tree.meta)
    meta.update({
        "selection_scores": sel["scores"],
        "criterion": criterion,
        "axis_weights": sel["axis_weights"],
        "precision_weight": precision_weight,
        "cost_weight": cost_weight,
        "stability_weight": stability_weight,
        "adaptive_multiobjective": True,
    })
    return WeightedMultiObjectiveWaveletPacket5DTree(nodes=tree.nodes, level=tree.level, meta=meta)



def weighted_multiobjective_wavelet_packet_5d_reconstruct(tree: WeightedMultiObjectiveWaveletPacket5DTree) -> list:
    proxy = AnisotropicWaveletPacket5DTree(nodes=tree.nodes, level=tree.level, meta={"families": tree.meta.get("families", ["haar"] * 5), "root_shape": tree.meta.get("root_shape", (0,0,0,0,0))})
    return anisotropic_wavelet_packet_5d_reconstruct(proxy)



def cross_branch_attentive_wavelet_packet_5d_decompose(
    tensor: list | np.ndarray,
    level: int = 1,
    candidate_families: list[str] | tuple[str, ...] | None = None,
    block_shape: tuple[int, int, int, int, int] = (1, 1, 2, 2, 2),
    criterion: str = "local_variation",
) -> CrossBranchAttentiveWaveletPacket5DTree:
    """Adapt node families while allowing attention across non-sibling branches at each depth."""
    level = ensure_positive_int(level, "level")
    x = _as_tensor5d(tensor)
    nodes={"": x.tolist()}
    current={"": x}
    node_families={"": ["haar"] * 5}
    branch_attention={}
    for depth_idx in range(level):
        next_candidates={}
        for path, ten in current.items():
            sel = select_wavelet_family_per_axis_5d_local_blocks(ten, candidate_families=candidate_families, block_shape=block_shape, criterion=criterion)
            families = tuple(sel["families"])
            node_families[path]=sel["families"]
            bands = _analysis_5d_anisotropic(ten, families)
            for band_name, band in bands.items():
                child = f"{path}/{band_name}" if path else band_name
                next_candidates[child] = np.asarray(band, dtype=np.float64)
                nodes[child] = band.tolist()
        # cross-branch attention among nodes at this new depth with different parents
        node_keys = list(next_candidates.keys())
        reps = np.stack([next_candidates[k].reshape(-1) for k in node_keys], axis=0)
        norms = np.linalg.norm(reps, axis=1, keepdims=True)
        sims = reps @ reps.T / np.maximum(norms @ norms.T, 1e-12)
        parent_keys = [k.rsplit('/', 1)[0] if '/' in k else '' for k in node_keys]
        mask = np.array([[1.0 if parent_keys[i] != parent_keys[j] else 0.0 for j in range(len(node_keys))] for i in range(len(node_keys))], dtype=np.float64)
        sims = sims * mask + (-1e9) * (1.0 - mask)
        attn = np.exp(sims - np.max(sims, axis=1, keepdims=True)) * mask
        denom = np.maximum(np.sum(attn, axis=1, keepdims=True), 1e-12)
        attn = attn / denom
        mixed = {}
        stacked = np.stack([next_candidates[k] for k in node_keys], axis=0)
        for i, key in enumerate(node_keys):
            if np.sum(mask[i]) == 0.0:
                mixed[key] = stacked[i]
            else:
                mixed[key] = 0.5 * stacked[i] + 0.5 * np.tensordot(attn[i], stacked, axes=(0, 0))
            nodes[key] = mixed[key].tolist()
        branch_attention[f"depth_{depth_idx+1}"] = {"nodes": node_keys, "weights": attn.tolist()}
        current = mixed
    return CrossBranchAttentiveWaveletPacket5DTree(nodes=nodes, level=level, meta={"root_shape": list(x.shape), "node_families": node_families, "branch_attention": branch_attention, "block_shape": list(block_shape), "criterion": criterion, "cross_branch_attention": True, "supported": available_wavelet_families()})



def cross_branch_attentive_wavelet_packet_5d_reconstruct(tree: CrossBranchAttentiveWaveletPacket5DTree) -> list:
    current = {k: np.asarray(v, dtype=np.float64) for k, v in tree.nodes.items() if k.count('/') + (1 if k else 0) == tree.level}
    band_names = [format(i, '05b').replace('0', 'l').replace('1', 'h') for i in range(32)]
    node_families = tree.meta.get('node_families', {})
    for _depth in range(tree.level - 1, -1, -1):
        nxt = {}
        prefixes = sorted(set(path.rsplit('/', 1)[0] if '/' in path else '' for path in current))
        for prefix in prefixes:
            base = f"{prefix}/" if prefix else ""
            bands = {name: current[base + name] for name in band_names}
            fams = tuple(node_families.get(prefix, ['haar'] * 5))
            nxt[prefix] = _synthesis_5d_anisotropic(bands, fams)
        current = nxt
    root_shape = tuple(tree.meta.get('root_shape', (0,0,0,0,0)))
    return current.get('', np.zeros(root_shape, dtype=np.float64)).tolist()



def _match_shape_mean_pool(x: np.ndarray, shape: tuple[int, int, int, int, int]) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    while y.shape != shape:
        slices = tuple(slice(0, min(y.shape[i], shape[i])) for i in range(5))
        y = y[slices]
        if y.shape == shape:
            break
        new = []
        for axis in range(5):
            if y.shape[axis] > shape[axis]:
                idx = np.linspace(0, y.shape[axis] - 1, shape[axis]).astype(int)
                y = np.take(y, idx, axis=axis)
        if y.shape == shape:
            break
        pad = [(0, max(0, shape[i] - y.shape[i])) for i in range(5)]
        y = np.pad(y, pad)
    return y



def learnable_multiobjective_weight_search_5d(
    tensor: list | np.ndarray,
    initial_axis_weights: list[float] | tuple[float, ...] | None = None,
    steps: int = 3,
    step_size: float = 0.1,
) -> list[float]:
    """Simple learnable-style search for axis weights from tensor statistics."""
    x = _as_tensor5d(tensor)
    weights = np.array(initial_axis_weights or [1.0] * 5, dtype=np.float64)
    if len(weights) != 5:
        raise ValueError("initial_axis_weights must have length 5")
    steps = ensure_positive_int(steps, "steps")
    axis_energy = np.array([float(np.mean(np.abs(np.diff(np.moveaxis(x, axis, 0), axis=0)))) if x.shape[axis] > 1 else 0.0 for axis in range(5)], dtype=np.float64)
    axis_energy = axis_energy / np.maximum(np.max(axis_energy), 1e-12)
    for _ in range(steps):
        weights = weights + step_size * (axis_energy - np.mean(axis_energy))
        weights = np.clip(weights, 0.1, None)
    return weights.tolist()



def learnable_multiobjective_wavelet_packet_5d_decompose(
    tensor: list | np.ndarray,
    level: int = 1,
    candidate_families: list[str] | tuple[str, ...] | None = None,
    block_shape: tuple[int, int, int, int, int] = (1, 1, 4, 4, 4),
    criterion: str = "local_variation",
    initial_axis_weights: list[float] | tuple[float, ...] | None = None,
    steps: int = 3,
    step_size: float = 0.1,
    precision_weight: float = 1.0,
    cost_weight: float = 0.05,
    stability_weight: float = 0.05,
) -> LearnableWeightedMultiObjectiveWaveletPacket5DTree:
    weights = learnable_multiobjective_weight_search_5d(tensor, initial_axis_weights=initial_axis_weights, steps=steps, step_size=step_size)
    tree = weighted_multiobjective_wavelet_packet_5d_decompose(
        tensor,
        level=level,
        candidate_families=candidate_families,
        block_shape=block_shape,
        criterion=criterion,
        axis_weights=weights,
        precision_weight=precision_weight,
        cost_weight=cost_weight,
        stability_weight=stability_weight,
    )
    meta = dict(tree.meta)
    meta.update({"learned_axis_weights": weights, "learnable_multiobjective": True, "steps": steps, "step_size": step_size})
    return LearnableWeightedMultiObjectiveWaveletPacket5DTree(nodes=tree.nodes, level=tree.level, meta=meta)



def learnable_multiobjective_wavelet_packet_5d_reconstruct(tree: LearnableWeightedMultiObjectiveWaveletPacket5DTree) -> list:
    proxy = WeightedMultiObjectiveWaveletPacket5DTree(nodes=tree.nodes, level=tree.level, meta=tree.meta)
    return weighted_multiobjective_wavelet_packet_5d_reconstruct(proxy)



def level_attentive_wavelet_packet_5d_decompose(
    tensor: list | np.ndarray,
    level: int = 1,
    candidate_families: list[str] | tuple[str, ...] | None = None,
    block_shape: tuple[int, int, int, int, int] = (1, 1, 2, 2, 2),
    criterion: str = "local_variation",
) -> LevelAttentiveWaveletPacket5DTree:
    """Allow attention from parent/root levels into deeper packet levels."""
    level = ensure_positive_int(level, "level")
    x = _as_tensor5d(tensor)
    nodes={"": x.tolist()}
    current={"": x}
    node_families={"": ["haar"] * 5}
    level_attention={}
    ancestor_tensors={"": x}
    for depth_idx in range(level):
        nxt={}
        for path, ten in current.items():
            sel = select_wavelet_family_per_axis_5d_local_blocks(ten, candidate_families=candidate_families, block_shape=block_shape, criterion=criterion)
            families = tuple(sel["families"])
            node_families[path] = sel["families"]
            bands = _analysis_5d_anisotropic(ten, families)
            for band_name, band in bands.items():
                child = f"{path}/{band_name}" if path else band_name
                parent_path = path
                root = ancestor_tensors[""]
                parent_tensor = ancestor_tensors[parent_path]
                child_arr = np.asarray(band, dtype=np.float64)
                parent_match = _match_shape_mean_pool(parent_tensor, child_arr.shape)
                root_match = _match_shape_mean_pool(root, child_arr.shape)
                reps = np.array([np.mean(np.abs(child_arr)), np.mean(np.abs(parent_match)), np.mean(np.abs(root_match))], dtype=np.float64)
                attn = np.exp(reps - np.max(reps))
                attn = attn / np.maximum(np.sum(attn), 1e-12)
                mixed = attn[0] * child_arr + attn[1] * parent_match + attn[2] * root_match
                nxt[child] = mixed
                nodes[child] = mixed.tolist()
                ancestor_tensors[child] = mixed
                level_attention[child] = attn.tolist()
        current = nxt
    return LevelAttentiveWaveletPacket5DTree(nodes=nodes, level=level, meta={"root_shape": list(x.shape), "node_families": node_families, "level_attention": level_attention, "block_shape": list(block_shape), "criterion": criterion, "level_attentive": True, "supported": available_wavelet_families()})



def level_attentive_wavelet_packet_5d_reconstruct(tree: LevelAttentiveWaveletPacket5DTree) -> list:
    current = {k: np.asarray(v, dtype=np.float64) for k, v in tree.nodes.items() if k.count('/') + (1 if k else 0) == tree.level}
    band_names = [format(i, '05b').replace('0', 'l').replace('1', 'h') for i in range(32)]
    node_families = tree.meta.get('node_families', {})
    for _depth in range(tree.level - 1, -1, -1):
        nxt = {}
        prefixes = sorted(set(path.rsplit('/', 1)[0] if '/' in path else '' for path in current))
        for prefix in prefixes:
            base = f"{prefix}/" if prefix else ""
            bands = {name: current[base + name] for name in band_names}
            fams = tuple(node_families.get(prefix, ['haar'] * 5))
            nxt[prefix] = _synthesis_5d_anisotropic(bands, fams)
        current = nxt
    root_shape = tuple(tree.meta.get('root_shape', (0,0,0,0,0)))
    return current.get('', np.zeros(root_shape, dtype=np.float64)).tolist()


__all__ = [
    "level_attentive_wavelet_packet_5d_reconstruct",
    "level_attentive_wavelet_packet_5d_decompose",
    "LevelAttentiveWaveletPacket5DTree",
    "learnable_multiobjective_wavelet_packet_5d_reconstruct",
    "learnable_multiobjective_wavelet_packet_5d_decompose",
    "learnable_multiobjective_weight_search_5d",
    "LearnableWeightedMultiObjectiveWaveletPacket5DTree",
    "weighted_multiobjective_wavelet_packet_5d_reconstruct",
    "weighted_multiobjective_wavelet_packet_5d_decompose",
    "weighted_multiobjective_select_wavelet_family_per_axis_5d",
    "WeightedMultiObjectiveWaveletPacket5DTree",
    "cross_branch_attentive_wavelet_packet_5d_reconstruct",
    "cross_branch_attentive_wavelet_packet_5d_decompose",
    "CrossBranchAttentiveWaveletPacket5DTree",
    "subband_attentive_wavelet_packet_5d_reconstruct",
    "subband_attentive_wavelet_packet_5d_decompose",
    "SubbandAttentiveWaveletPacket5DTree",
    "regularized_adaptive_wavelet_packet_5d_reconstruct",
    "regularized_adaptive_wavelet_packet_5d_decompose",
    "regularized_select_wavelet_family_per_axis_5d",
    "RegularizedAdaptiveWaveletPacket5DTree",
    "subband_adaptive_wavelet_packet_5d_reconstruct",
    "subband_adaptive_wavelet_packet_5d_decompose",
    "SubbandAdaptiveWaveletPacket5DTree",
    "spatially_variable_wavelet_packet_5d_reconstruct",
    "spatially_variable_wavelet_packet_5d_decompose",
    "SpatiallyAdaptiveWaveletPacket5DTree",
    "adaptive_blockwise_wavelet_packet_5d_reconstruct",
    "adaptive_blockwise_wavelet_packet_5d_decompose",
    "select_wavelet_family_per_axis_5d_local_blocks",
    "BlockAdaptiveWaveletPacket5DTree",
    "adaptive_wavelet_packet_5d_reconstruct",
    "adaptive_wavelet_packet_5d_decompose",
    "select_wavelet_family_per_axis_5d",
    "AdaptiveWaveletPacket5DTree",
    "WaveletPacket5DTree",
    "AnisotropicWaveletPacket5DTree",
    "wavelet_packet_5d_decompose",
    "wavelet_packet_5d_reconstruct",
    "anisotropic_wavelet_packet_5d_decompose",
    "anisotropic_wavelet_packet_5d_reconstruct",
]
