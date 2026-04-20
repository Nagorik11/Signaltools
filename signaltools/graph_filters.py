"""Graph signal filtering utilities."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .utils import to_1d_float_array


def graph_laplacian(adjacency: list[list[float]] | np.ndarray, normalized: bool = True) -> np.ndarray:
    A = np.asarray(adjacency, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    degree = np.sum(A, axis=1)
    L = np.diag(degree) - A
    if not normalized:
        return L
    inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
    return inv_sqrt @ L @ inv_sqrt


def graph_fourier_basis(adjacency: list[list[float]] | np.ndarray, normalized: bool = True) -> tuple[np.ndarray, np.ndarray]:
    L = graph_laplacian(adjacency, normalized=normalized)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    return eigenvalues, eigenvectors


def graph_filter_signal(
    signal: list[float] | list[int],
    adjacency: list[list[float]] | np.ndarray,
    response: Callable[[np.ndarray], np.ndarray] | None = None,
    normalized: bool = True,
) -> list[float]:
    x = to_1d_float_array(signal)
    eigenvalues, eigenvectors = graph_fourier_basis(adjacency, normalized=normalized)
    if len(x) != eigenvectors.shape[0]:
        raise ValueError("signal length must match graph size")
    if response is None:
        response = lambda lam: 1.0 / (1.0 + lam)
    H = np.asarray(response(eigenvalues), dtype=np.float64)
    coeffs = eigenvectors.T @ x
    y = eigenvectors @ (H * coeffs)
    return y.astype(np.float64).tolist()


def graph_polynomial_filter(
    signal: list[float] | list[int],
    adjacency: list[list[float]] | np.ndarray,
    coeffs: list[float],
    normalized: bool = True,
) -> list[float]:
    x = to_1d_float_array(signal)
    L = graph_laplacian(adjacency, normalized=normalized)
    if len(x) != L.shape[0]:
        raise ValueError("signal length must match graph size")
    y = np.zeros_like(x)
    current = np.array(x, copy=True)
    for k, coeff in enumerate(coeffs):
        if k == 0:
            current = x
        elif k == 1:
            current = L @ x
        else:
            current = L @ current
        y += float(coeff) * current
    return y.astype(np.float64).tolist()


__all__ = ["graph_laplacian", "graph_fourier_basis", "graph_filter_signal", "graph_polynomial_filter"]
