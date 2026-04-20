"""Basic grayscale 2D morphology."""

from __future__ import annotations

import numpy as np


def _as_2d(image: list[list[float]] | np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("image must be 2D")
    return x


def _validate_kernel(kernel_size: int) -> int:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    return kernel_size


def _sliding_extreme(image: np.ndarray, kernel_size: int, mode: str, extreme: str) -> np.ndarray:
    kernel_size = _validate_kernel(kernel_size)
    pad = kernel_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode=mode)
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i : i + kernel_size, j : j + kernel_size]
            out[i, j] = np.max(window) if extreme == "max" else np.min(window)
    return out


def dilation_2d(image: list[list[float]] | np.ndarray, kernel_size: int = 3, mode: str = "edge") -> list[list[float]]:
    return _sliding_extreme(_as_2d(image), kernel_size, mode, "max").tolist()


def erosion_2d(image: list[list[float]] | np.ndarray, kernel_size: int = 3, mode: str = "edge") -> list[list[float]]:
    return _sliding_extreme(_as_2d(image), kernel_size, mode, "min").tolist()


def opening_2d(image: list[list[float]] | np.ndarray, kernel_size: int = 3, mode: str = "edge") -> list[list[float]]:
    eroded = np.asarray(erosion_2d(image, kernel_size, mode), dtype=np.float64)
    return dilation_2d(eroded, kernel_size, mode)


def closing_2d(image: list[list[float]] | np.ndarray, kernel_size: int = 3, mode: str = "edge") -> list[list[float]]:
    dilated = np.asarray(dilation_2d(image, kernel_size, mode), dtype=np.float64)
    return erosion_2d(dilated, kernel_size, mode)


def median_filter_2d(image: list[list[float]] | np.ndarray, kernel_size: int = 3, mode: str = "edge") -> list[list[float]]:
    x = _as_2d(image)
    kernel_size = _validate_kernel(kernel_size)
    pad = kernel_size // 2
    padded = np.pad(x, ((pad, pad), (pad, pad)), mode=mode)
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = np.median(padded[i : i + kernel_size, j : j + kernel_size])
    return out.tolist()


def morphological_gradient_2d(image: list[list[float]] | np.ndarray, kernel_size: int = 3, mode: str = "edge") -> list[list[float]]:
    dil = np.asarray(dilation_2d(image, kernel_size, mode), dtype=np.float64)
    ero = np.asarray(erosion_2d(image, kernel_size, mode), dtype=np.float64)
    return (dil - ero).tolist()


__all__ = ["dilation_2d", "erosion_2d", "opening_2d", "closing_2d", "median_filter_2d", "morphological_gradient_2d", "dilation_3d", "erosion_3d", "opening_3d", "closing_3d", "median_filter_3d", "morphological_gradient_3d", "dilation_3d_kernel", "erosion_3d_kernel", "opening_3d_kernel", "closing_3d_kernel"]


def _as_3d(volume: list[list[list[float]]] | np.ndarray) -> np.ndarray:
    x = np.asarray(volume, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("volume must be 3D")
    return x


def _sliding_extreme_3d(volume: np.ndarray, kernel_size: int, mode: str, extreme: str) -> np.ndarray:
    kernel_size = _validate_kernel(kernel_size)
    pad = kernel_size // 2
    padded = np.pad(volume, ((pad, pad), (pad, pad), (pad, pad)), mode=mode)
    out = np.zeros_like(volume)
    for z in range(volume.shape[0]):
        for i in range(volume.shape[1]):
            for j in range(volume.shape[2]):
                window = padded[z : z + kernel_size, i : i + kernel_size, j : j + kernel_size]
                out[z, i, j] = np.max(window) if extreme == "max" else np.min(window)
    return out


def dilation_3d(volume: list[list[list[float]]] | np.ndarray, kernel_size: int = 3, mode: str = "edge") -> list[list[list[float]]]:
    return _sliding_extreme_3d(_as_3d(volume), kernel_size, mode, "max").tolist()


def erosion_3d(volume: list[list[list[float]]] | np.ndarray, kernel_size: int = 3, mode: str = "edge") -> list[list[list[float]]]:
    return _sliding_extreme_3d(_as_3d(volume), kernel_size, mode, "min").tolist()


def opening_3d(volume: list[list[list[float]]] | np.ndarray, kernel_size: int = 3, mode: str = "edge") -> list[list[list[float]]]:
    eroded = np.asarray(erosion_3d(volume, kernel_size, mode), dtype=np.float64)
    return dilation_3d(eroded, kernel_size, mode)


def closing_3d(volume: list[list[list[float]]] | np.ndarray, kernel_size: int = 3, mode: str = "edge") -> list[list[list[float]]]:
    dilated = np.asarray(dilation_3d(volume, kernel_size, mode), dtype=np.float64)
    return erosion_3d(dilated, kernel_size, mode)


def median_filter_3d(volume: list[list[list[float]]] | np.ndarray, kernel_size: int = 3, mode: str = "edge") -> list[list[list[float]]]:
    x = _as_3d(volume)
    kernel_size = _validate_kernel(kernel_size)
    pad = kernel_size // 2
    padded = np.pad(x, ((pad, pad), (pad, pad), (pad, pad)), mode=mode)
    out = np.zeros_like(x)
    for z in range(x.shape[0]):
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                out[z, i, j] = np.median(padded[z : z + kernel_size, i : i + kernel_size, j : j + kernel_size])
    return out.tolist()


def morphological_gradient_3d(volume: list[list[list[float]]] | np.ndarray, kernel_size: int = 3, mode: str = "edge") -> list[list[list[float]]]:
    dil = np.asarray(dilation_3d(volume, kernel_size, mode), dtype=np.float64)
    ero = np.asarray(erosion_3d(volume, kernel_size, mode), dtype=np.float64)
    return (dil - ero).tolist()



def _kernel_offsets(kernel: np.ndarray) -> np.ndarray:
    center = np.array(kernel.shape) // 2
    coords = np.argwhere(kernel > 0)
    return coords - center


def dilation_3d_kernel(volume: list[list[list[float]]] | np.ndarray, kernel: list[list[list[float]]] | np.ndarray, mode: str = "edge") -> list[list[list[float]]]:
    vol = _as_3d(volume)
    ker = np.asarray(kernel, dtype=np.float64)
    if ker.ndim != 3:
        raise ValueError("kernel must be 3D")
    offsets = _kernel_offsets(ker)
    pad = tuple((s // 2, s // 2) for s in ker.shape)
    padded = np.pad(vol, pad, mode=mode)
    out = np.zeros_like(vol)
    center = np.array(ker.shape) // 2
    for z in range(vol.shape[0]):
        for i in range(vol.shape[1]):
            for j in range(vol.shape[2]):
                values = []
                for dz, di, dj in offsets:
                    values.append(padded[z + dz + center[0], i + di + center[1], j + dj + center[2]])
                out[z, i, j] = np.max(values)
    return out.tolist()


def erosion_3d_kernel(volume: list[list[list[float]]] | np.ndarray, kernel: list[list[list[float]]] | np.ndarray, mode: str = "edge") -> list[list[list[float]]]:
    vol = _as_3d(volume)
    ker = np.asarray(kernel, dtype=np.float64)
    if ker.ndim != 3:
        raise ValueError("kernel must be 3D")
    offsets = _kernel_offsets(ker)
    pad = tuple((s // 2, s // 2) for s in ker.shape)
    padded = np.pad(vol, pad, mode=mode)
    out = np.zeros_like(vol)
    center = np.array(ker.shape) // 2
    for z in range(vol.shape[0]):
        for i in range(vol.shape[1]):
            for j in range(vol.shape[2]):
                values = []
                for dz, di, dj in offsets:
                    values.append(padded[z + dz + center[0], i + di + center[1], j + dj + center[2]])
                out[z, i, j] = np.min(values)
    return out.tolist()


def opening_3d_kernel(volume: list[list[list[float]]] | np.ndarray, kernel: list[list[list[float]]] | np.ndarray, mode: str = "edge") -> list[list[list[float]]]:
    return dilation_3d_kernel(erosion_3d_kernel(volume, kernel, mode), kernel, mode)


def closing_3d_kernel(volume: list[list[list[float]]] | np.ndarray, kernel: list[list[list[float]]] | np.ndarray, mode: str = "edge") -> list[list[list[float]]]:
    return erosion_3d_kernel(dilation_3d_kernel(volume, kernel, mode), kernel, mode)
