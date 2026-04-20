"""Graph positional encoding helpers."""

from __future__ import annotations

import numpy as np

from .graph_filters import graph_laplacian
from .utils import ensure_positive_int



def laplacian_positional_encoding(
    adjacency: list[list[float]] | np.ndarray,
    dimensions: int = 4,
    normalized: bool = True,
) -> list[list[float]]:
    """Return the smallest non-trivial Laplacian eigenvectors as positional encodings."""
    dimensions = ensure_positive_int(dimensions, "dimensions")
    L = graph_laplacian(adjacency, normalized=normalized)
    vals, vecs = np.linalg.eigh(L)
    order = np.argsort(vals)
    vecs = vecs[:, order]
    start = 1 if vecs.shape[1] > 1 else 0
    stop = min(start + dimensions, vecs.shape[1])
    enc = vecs[:, start:stop]
    return enc.tolist()



def random_walk_positional_encoding(
    adjacency: list[list[float]] | np.ndarray,
    steps: int = 4,
) -> list[list[float]]:
    """Return powers of the random-walk matrix diagonal/off-diagonal summaries."""
    steps = ensure_positive_int(steps, "steps")
    A = np.asarray(adjacency, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    degrees = np.sum(A, axis=1, keepdims=True)
    P = A / np.maximum(degrees, 1e-12)
    current = np.eye(A.shape[0], dtype=np.float64)
    encodings = []
    for _ in range(steps):
        current = current @ P
        encodings.append(np.diag(current))
    return np.stack(encodings, axis=1).tolist()



def augment_with_graph_positional_encoding(
    node_features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    method: str = "laplacian",
    dimensions: int = 4,
    steps: int = 4,
) -> list[list[float]]:
    """Concatenate node features with positional encodings."""
    x = np.asarray(node_features, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("node_features must be 2D")
    if method == "laplacian":
        pe = np.asarray(laplacian_positional_encoding(adjacency, dimensions=dimensions), dtype=np.float64)
    elif method == "random_walk":
        pe = np.asarray(random_walk_positional_encoding(adjacency, steps=steps), dtype=np.float64)
    else:
        raise ValueError("Unsupported positional encoding method")
    if pe.shape[0] != x.shape[0]:
        raise ValueError("node count must match graph size")
    return np.concatenate([x, pe], axis=1).tolist()


__all__ = [
    "laplacian_positional_encoding",
    "random_walk_positional_encoding",
    "augment_with_graph_positional_encoding",
]
