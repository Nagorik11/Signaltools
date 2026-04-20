"""Graph wavelets and Chebyshev graph filtering."""

from __future__ import annotations

import math
import numpy as np

from .graph_filters import graph_fourier_basis, graph_laplacian
from .utils import to_1d_float_array


def chebyshev_graph_filter(
    signal: list[float] | list[int],
    adjacency: list[list[float]] | np.ndarray,
    coeffs: list[float],
    normalized: bool = True,
) -> list[float]:
    x = to_1d_float_array(signal)
    L = graph_laplacian(adjacency, normalized=normalized)
    if len(x) != L.shape[0]:
        raise ValueError("signal length must match graph size")
    if not coeffs:
        return [0.0 for _ in x]
    lmax = float(np.max(np.linalg.eigvalsh(L)))
    if lmax <= 0:
        return x.tolist()
    Lt = (2.0 / lmax) * L - np.eye(L.shape[0])
    T0 = x.copy()
    y = float(coeffs[0]) * T0
    if len(coeffs) == 1:
        return y.tolist()
    T1 = Lt @ x
    y = y + float(coeffs[1]) * T1
    for k in range(2, len(coeffs)):
        T2 = 2.0 * (Lt @ T1) - T0
        y = y + float(coeffs[k]) * T2
        T0, T1 = T1, T2
    return y.tolist()


def graph_wavelet_kernel(lambdas: np.ndarray, scale: float, kind: str = "heat") -> np.ndarray:
    if kind == "heat":
        return np.exp(-scale * lambdas)
    if kind == "mexican_hat":
        return scale * lambdas * np.exp(-scale * lambdas)
    raise ValueError("Unsupported graph wavelet kernel")


def graph_wavelet_transform(
    signal: list[float] | list[int],
    adjacency: list[list[float]] | np.ndarray,
    scales: list[float],
    normalized: bool = True,
    kind: str = "heat",
) -> list[list[float]]:
    x = to_1d_float_array(signal)
    lambdas, U = graph_fourier_basis(adjacency, normalized=normalized)
    if len(x) != U.shape[0]:
        raise ValueError("signal length must match graph size")
    coeffs = U.T @ x
    outputs: list[list[float]] = []
    for scale in scales:
        kernel = graph_wavelet_kernel(lambdas, float(scale), kind=kind)
        outputs.append((U @ (kernel * coeffs)).astype(np.float64).tolist())
    return outputs


__all__ = ["chebyshev_graph_filter", "graph_wavelet_kernel", "graph_wavelet_transform"]
