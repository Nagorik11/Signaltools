from __future__ import annotations

import numpy as np


def generate_tensor_volume(
    shape: tuple = (20, 16, 12),
    fiber_dir: tuple = (0.0, 1.0, 0.0),
    fa_val: float = 0.7,
    md_val: float = 0.0007,
    noise: float = 0.02,
) -> np.ndarray:
    Z, Y, X = shape
    dir_vec = np.asarray(fiber_dir, dtype=np.float64)
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)

    ev1 = dir_vec
    rng = np.random.default_rng(42)
    ev2 = np.array([1, 0, 0], dtype=np.float64)
    if abs(np.dot(ev1, ev2)) > 0.9:
        ev2 = np.array([0, 0, 1], dtype=np.float64)
    ev2 = ev2 - np.dot(ev2, ev1) * ev1
    ev2 = ev2 / np.linalg.norm(ev2)
    ev3 = np.cross(ev1, ev2)

    l1 = md_val * (1 + 2 * fa_val / np.sqrt(3 - 2 * fa_val ** 2))
    l2 = md_val * (1 - fa_val / np.sqrt(3 - 2 * fa_val ** 2))
    l3 = l2

    tensors = np.zeros((Z, Y, X, 3, 3))
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                center = np.array([Z // 2, Y // 2, X // 2])
                d = abs(z - center[0]) + abs(y - center[1]) + abs(x - center[2])
                if d > min(Z, Y, X) * 1.0:
                    tensors[z, y, x] = np.diag([l3, l3, l3])
                    continue
                T = (l1 * np.outer(ev1, ev1) +
                     l2 * np.outer(ev2, ev2) +
                     l3 * np.outer(ev3, ev3))
                tensors[z, y, x] = T + rng.normal(0, noise * md_val, (3, 3))
                tensors[z, y, x] = (tensors[z, y, x] + tensors[z, y, x].T) / 2

    return tensors


def generate_crossing_fiber(
    shape: tuple = (20, 16, 12),
    fa_val: float = 0.6,
    md_val: float = 0.0007,
    angle: float = 60.0,
) -> np.ndarray:
    Z, Y, X = shape
    half_angle = np.radians(angle / 2)
    dir1 = np.array([0.0, np.cos(half_angle), np.sin(half_angle)], dtype=np.float64)
    dir2 = np.array([0.0, np.cos(half_angle), -np.sin(half_angle)], dtype=np.float64)

    l1 = md_val * (1 + 2 * fa_val / np.sqrt(3 - 2 * fa_val ** 2))
    l2 = md_val * (1 - fa_val / np.sqrt(3 - 2 * fa_val ** 2))
    l3 = l2

    def make_tensor(dir_vec):
        ev1 = dir_vec / np.linalg.norm(dir_vec)
        return (l1 * np.outer(ev1, ev1) +
                l2 * np.outer(dir_vec_orthogonal(ev1), dir_vec_orthogonal(ev1)) +
                l3 * np.outer(np.cross(ev1, dir_vec_orthogonal(ev1)),
                              np.cross(ev1, dir_vec_orthogonal(ev1))))

    def dir_vec_orthogonal(v):
        if abs(v[2]) < 0.9:
            return np.array([0, 0, 1], dtype=np.float64)
        return np.array([1, 0, 0], dtype=np.float64)

    T1 = make_tensor(dir1)
    T2 = make_tensor(dir2)

    tensors = np.zeros((Z, Y, X, 3, 3))
    cx, cy, cz = X // 2, Y // 2, Z // 2
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                alpha = np.exp(-((x - cx) ** 2) / (2 * 8))
                beta = 1 - alpha
                tensors[z, y, x] = alpha * T1 + beta * T2

    return tensors


def generate_synthetic_dwi(
    shape: tuple = (8, 8, 4),
    n_directions: int = 30,
    bval: float = 1000.0,
) -> tuple:
    gradients = _gradient_sampling(n_directions)
    tensors = generate_tensor_volume(shape)
    Z, Y, X = shape

    S0 = 1000.0
    dw_signals = np.zeros((n_directions, Z, Y, X))
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                D = tensors[z, y, x]
                for i in range(n_directions):
                    g = gradients[i]
                    b = bval * np.dot(g, D @ g)
                    dw_signals[i, z, y, x] = S0 * np.exp(-b)

    return dw_signals, gradients, tensors


def _gradient_sampling(n: int) -> np.ndarray:
    rng = np.random.default_rng(123)
    g = rng.normal(size=(n, 3))
    norms = np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
    return g / norms
