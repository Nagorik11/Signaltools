"""Wavelet packet filter-bank helpers with several orthogonal and biorthogonal families."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .utils import ensure_positive_int, to_1d_float_array


@dataclass
class WaveletPacketTree:
    nodes: dict[str, list[float]]
    level: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Each entry defines low-pass decomposition and reconstruction filters.
# Orthogonal families use the same prototype for analysis and synthesis.
_WAVELETS: dict[str, dict[str, list[float]]] = {
    "haar": {
        "dec_lo": [0.7071067811865476, 0.7071067811865476],
        "rec_lo": [0.7071067811865476, 0.7071067811865476],
        "kind": "orthogonal",
    },
    "db2": {
        "dec_lo": [0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604],
        "rec_lo": [0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604],
        "kind": "orthogonal",
    },
    "db4": {
        "dec_lo": [-0.010597401785069032, 0.0328830116668852, 0.030841381835560764, -0.18703481171909309, -0.027983769416859854, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523],
        "rec_lo": [-0.010597401785069032, 0.0328830116668852, 0.030841381835560764, -0.18703481171909309, -0.027983769416859854, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523],
        "kind": "orthogonal",
    },
    "db6": {
        "dec_lo": [-0.0010773010849955799, 0.004777257510945511, 0.0005538422009938016, -0.03158203931748603, 0.027522865530305727, 0.09750160558732248, -0.1297668675672625, -0.22626469396544, 0.3152503517091982, 0.7511339080215775, 0.49462389039845306, 0.11154074335008017],
        "rec_lo": [-0.0010773010849955799, 0.004777257510945511, 0.0005538422009938016, -0.03158203931748603, 0.027522865530305727, 0.09750160558732248, -0.1297668675672625, -0.22626469396544, 0.3152503517091982, 0.7511339080215775, 0.49462389039845306, 0.11154074335008017],
        "kind": "orthogonal",
    },
    "db8": {
        "dec_lo": [-0.00011747678412476953, 0.0006754494064505693, -0.00039174037337694705, -0.004870352993451574, 0.008746094047405777, 0.013981027917398282, -0.044088253930794755, -0.017369301001807547, 0.12874742662047847, 0.0004724845739132828, -0.2840155429615469, -0.015829105256349306, 0.5853546836548691, 0.6756307362972898, 0.31287159091429995, 0.05441584224308161],
        "rec_lo": [-0.00011747678412476953, 0.0006754494064505693, -0.00039174037337694705, -0.004870352993451574, 0.008746094047405777, 0.013981027917398282, -0.044088253930794755, -0.017369301001807547, 0.12874742662047847, 0.0004724845739132828, -0.2840155429615469, -0.015829105256349306, 0.5853546836548691, 0.6756307362972898, 0.31287159091429995, 0.05441584224308161],
        "kind": "orthogonal",
    },
    "sym2": {
        "dec_lo": [-0.12940952255126037, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
        "rec_lo": [-0.12940952255126037, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
        "kind": "orthogonal",
    },
    "sym4": {
        "dec_lo": [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
        "rec_lo": [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
        "kind": "orthogonal",
    },
    "sym6": {
        "dec_lo": [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585632995, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148],
        "rec_lo": [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585632995, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148],
        "kind": "orthogonal",
    },
    "coif1": {
        "dec_lo": [-0.015655728135791993, -0.0727326195128539, 0.3848648468648578, 0.8525720202122554, 0.3378976624574818, -0.0727326195128539],
        "rec_lo": [-0.015655728135791993, -0.0727326195128539, 0.3848648468648578, 0.8525720202122554, 0.3378976624574818, -0.0727326195128539],
        "kind": "orthogonal",
    },
    "coif2": {
        "dec_lo": [-0.0007205494453645122, -0.0018232088707029932, 0.0056114348193944995, 0.023680171946334084, -0.0594344186464569, -0.0764885990783064, 0.41700518442169254, 0.8127236354455423, 0.3861100668211622, -0.06737255472196302, -0.04146493678175915, 0.016387336463522112],
        "rec_lo": [-0.0007205494453645122, -0.0018232088707029932, 0.0056114348193944995, 0.023680171946334084, -0.0594344186464569, -0.0764885990783064, 0.41700518442169254, 0.8127236354455423, 0.3861100668211622, -0.06737255472196302, -0.04146493678175915, 0.016387336463522112],
        "kind": "orthogonal",
    },
    "coif3": {
        "dec_lo": [-3.459977283621255e-05, -7.098330313814125e-05, 0.0004662169601128863, 0.0011175187708906016, -0.0025745176887502236, -0.00900797613666158, 0.015880544863669452, 0.03455502757306163, -0.08230192710688598, -0.07179982161931202, 0.4284834763776168, 0.7937772226256206, 0.405176902409615, -0.06112339000267287, -0.0657719112818555, 0.023452696141836267, 0.007782596427325418, -0.003793512864491014],
        "rec_lo": [-3.459977283621255e-05, -7.098330313814125e-05, 0.0004662169601128863, 0.0011175187708906016, -0.0025745176887502236, -0.00900797613666158, 0.015880544863669452, 0.03455502757306163, -0.08230192710688598, -0.07179982161931202, 0.4284834763776168, 0.7937772226256206, 0.405176902409615, -0.06112339000267287, -0.0657719112818555, 0.023452696141836267, 0.007782596427325418, -0.003793512864491014],
        "kind": "orthogonal",
    },
    "bior53": {
        "dec_lo": [-0.125, 0.25, 0.75, 0.25, -0.125],
        "rec_lo": [0.5, 1.0, 0.5],
        "kind": "biorthogonal",
    },
    "bior97": {
        "dec_lo": [0.02674875741080976, -0.01686411844287495, -0.07822326652898785, 0.2668641184428723, 0.6029490182363579, 0.2668641184428723, -0.07822326652898785, -0.01686411844287495, 0.02674875741080976],
        "rec_lo": [0.0, 0.09127176311424948, -0.05754352622849957, -0.591271763114247, 1.115087052456994, -0.591271763114247, -0.05754352622849957, 0.09127176311424948, 0.0],
        "kind": "biorthogonal",
    },
    "cdf53": {
        "dec_lo": [-0.125, 0.25, 0.75, 0.25, -0.125],
        "rec_lo": [0.5, 1.0, 0.5],
        "kind": "biorthogonal",
    },
    "cdf97": {
        "dec_lo": [0.02674875741080976, -0.01686411844287495, -0.07822326652898785, 0.2668641184428723, 0.6029490182363579, 0.2668641184428723, -0.07822326652898785, -0.01686411844287495, 0.02674875741080976],
        "rec_lo": [0.0, 0.09127176311424948, -0.05754352622849957, -0.591271763114247, 1.115087052456994, -0.591271763114247, -0.05754352622849957, 0.09127176311424948, 0.0],
        "kind": "biorthogonal",
    },
    "spline53": {
        "dec_lo": [-0.125, 0.25, 0.75, 0.25, -0.125],
        "rec_lo": [0.5, 1.0, 0.5],
        "kind": "biorthogonal",
    },
    "spline97": {
        "dec_lo": [0.02674875741080976, -0.01686411844287495, -0.07822326652898785, 0.2668641184428723, 0.6029490182363579, 0.2668641184428723, -0.07822326652898785, -0.01686411844287495, 0.02674875741080976],
        "rec_lo": [0.0, 0.09127176311424948, -0.05754352622849957, -0.591271763114247, 1.115087052456994, -0.591271763114247, -0.05754352622849957, 0.09127176311424948, 0.0],
        "kind": "biorthogonal",
    },
    "bior22": {
        "dec_lo": [0.0, -0.1767766952966369, 0.3535533905932738, 1.0606601717798212, 0.3535533905932738, -0.1767766952966369],
        "rec_lo": [0.0, 0.3535533905932738, 0.7071067811865476, 0.3535533905932738, 0.0, 0.0],
        "kind": "biorthogonal",
    },
    "bior44": {
        "dec_lo": [0.0, 0.03782845550726404, -0.023849465019556843, -0.11062440441843718, 0.37740285561283066, 0.8526986790088938, 0.37740285561283066, -0.11062440441843718, -0.023849465019556843, 0.03782845550726404],
        "rec_lo": [0.0, -0.06453888262869706, -0.04068941760916406, 0.41809227322161724, 0.7884856164055829, 0.41809227322161724, -0.04068941760916406, -0.06453888262869706, 0.0, 0.0],
        "kind": "biorthogonal",
    },
}


def available_wavelet_families() -> list[str]:
    """Return the supported wavelet family names."""
    return sorted(_WAVELETS)



def wavelet_family_kind(family: str) -> str:
    """Return whether a family is orthogonal or biorthogonal."""
    if family not in _WAVELETS:
        raise ValueError("Unsupported wavelet family")
    return str(_WAVELETS[family]["kind"])



def wavelet_filters(family: str = "haar") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return analysis and synthesis filters for a family.

    Returns:
        (h0, h1, g0, g1): analysis low/high and synthesis low/high filters.
    """
    if family not in _WAVELETS:
        raise ValueError("Unsupported wavelet family")
    entry = _WAVELETS[family]
    h0 = np.asarray(entry["dec_lo"], dtype=np.float64)
    g0 = np.asarray(entry["rec_lo"], dtype=np.float64)
    h1 = np.array([((-1) ** k) * g0[::-1][k] for k in range(len(g0))], dtype=np.float64)
    g1 = np.array([((-1) ** (k + 1)) * h0[::-1][k] for k in range(len(h0))], dtype=np.float64)
    return h0, h1, g0, g1



def _analysis_pair(signal: list[float] | list[int], family: str) -> tuple[list[float], list[float]]:
    x = to_1d_float_array(signal)
    h0, h1, _, _ = wavelet_filters(family)
    low = np.convolve(x, h0, mode="same")[::2]
    high = np.convolve(x, h1, mode="same")[::2]
    return low.tolist(), high.tolist()



def _synthesis_pair(low: list[float] | list[int], high: list[float] | list[int], family: str) -> list[float]:
    l = to_1d_float_array(low, name="low")
    h = to_1d_float_array(high, name="high")
    _, _, g0, g1 = wavelet_filters(family)
    n = max(len(l), len(h))
    up_l = np.zeros(2 * n, dtype=np.float64)
    up_h = np.zeros(2 * n, dtype=np.float64)
    up_l[::2][: len(l)] = l
    up_h[::2][: len(h)] = h
    return (np.convolve(up_l, g0, mode="same") + np.convolve(up_h, g1, mode="same")).tolist()



def wavelet_packet_decompose(signal: list[float] | list[int], level: int = 3, family: str = "haar") -> WaveletPacketTree:
    """Decompose a signal into a full wavelet packet tree."""
    level = ensure_positive_int(level, "level")
    x = to_1d_float_array(signal)
    nodes: dict[str, list[float]] = {"": x.tolist()}
    current = {"": x.tolist()}
    for _ in range(level):
        nxt: dict[str, list[float]] = {}
        for path, values in current.items():
            low, high = _analysis_pair(values, family)
            nxt[path + "a"] = low
            nxt[path + "d"] = high
            nodes[path + "a"] = low
            nodes[path + "d"] = high
        current = nxt
    return WaveletPacketTree(
        nodes=nodes,
        level=level,
        meta={
            "wavelet": family,
            "kind": wavelet_family_kind(family),
            "supported": available_wavelet_families(),
        },
    )



def wavelet_packet_reconstruct(tree: WaveletPacketTree) -> list[float]:
    """Reconstruct the root signal from a packet tree."""
    family = tree.meta.get("wavelet", "haar")
    current = {k: v for k, v in tree.nodes.items() if len(k) == tree.level}
    for _depth in range(tree.level - 1, -1, -1):
        nxt: dict[str, list[float]] = {}
        prefixes = sorted(set(path[:-1] for path in current))
        for prefix in prefixes:
            nxt[prefix] = _synthesis_pair(current[prefix + "a"], current[prefix + "d"], family)
        current = nxt
    return current.get("", [])


__all__ = [
    "WaveletPacketTree",
    "available_wavelet_families",
    "wavelet_family_kind",
    "wavelet_filters",
    "wavelet_packet_decompose",
    "wavelet_packet_reconstruct",
]
