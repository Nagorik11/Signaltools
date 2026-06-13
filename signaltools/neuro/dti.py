from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DiffusionTensor:
    tensor: np.ndarray  # 3x3 symmetric
    eigenvalues: np.ndarray  # (3,)
    eigenvectors: np.ndarray  # 3x3
    fa: float = 0.0
    md: float = 0.0
    ad: float = 0.0
    rd: float = 0.0
    cl: float = 0.0
    cp: float = 0.0
    cs: float = 0.0
    mo: float = 0.0
    vr: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tensor": self.tensor.tolist(),
            "eigenvalues": [round(float(v), 6) for v in self.eigenvalues],
            "eigenvectors": self.eigenvectors.tolist(),
            "fa": round(self.fa, 6),
            "md": round(self.md, 6),
            "ad": round(self.ad, 6),
            "rd": round(self.rd, 6),
            "cl": round(self.cl, 6),
            "cp": round(self.cp, 6),
            "cs": round(self.cs, 6),
            "mo": round(self.mo, 6),
            "vr": round(self.vr, 6),
        }


def _design_matrix(gradients: np.ndarray, bval: float) -> np.ndarray:
    gx = gradients[:, 0]
    gy = gradients[:, 1]
    gz = gradients[:, 2]
    return np.column_stack([
        gx * gx,
        gy * gy,
        gz * gz,
        2 * gx * gy,
        2 * gx * gz,
        2 * gy * gz,
    ]) * (-bval)


def _fractional_anisotropy(ev: np.ndarray) -> float:
    mean = ev.mean()
    num = np.sqrt(((ev - mean) ** 2).sum())
    den = np.sqrt((ev ** 2).sum())
    if den < 1e-12:
        return 0.0
    return float(np.sqrt(1.5) * num / den)


def _shape_indices(ev: np.ndarray) -> Tuple[float, float, float]:
    l1, l2, l3 = ev[0], ev[1], ev[2]
    den = l1 + l2 + l3
    if den < 1e-12:
        return 0.0, 0.0, 0.0
    cl = (l1 - l2) / den
    cp = 2 * (l2 - l3) / den
    cs = 3 * l3 / den
    return float(cl), float(cp), float(cs)


def _mode_anisotropy(ev: np.ndarray) -> float:
    l1, l2, l3 = ev[0], ev[1], ev[2]
    den = l1 + l2 + l3
    if den < 1e-12:
        return 0.0
    return float(2 * (l2 - l1 - l3) / den)


def _volume_ratio(ev: np.ndarray) -> float:
    l1, l2, l3 = ev[0], ev[1], ev[2]
    mean = ev.mean()
    if mean < 1e-12:
        return 1.0
    return float((l1 * l2 * l3) / (mean ** 3))


def fit_tensor(
    dw_signals: np.ndarray,
    b0: np.ndarray,
    gradients: np.ndarray,
    bval: float = 1000.0,
) -> DiffusionTensor:
    nobs = dw_signals.shape[0]
    if nobs < 6:
        raise ValueError("Need at least 6 diffusion-weighted measurements")
    S = np.asarray(dw_signals, dtype=np.float64)
    S0 = np.asarray(b0, dtype=np.float64).mean()
    y = -np.log(np.maximum(S / (S0 + 1e-12), 1e-12))
    A = _design_matrix(np.asarray(gradients, dtype=np.float64), bval)
    Dvec, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    D = np.array([
        [Dvec[0], Dvec[3], Dvec[4]],
        [Dvec[3], Dvec[1], Dvec[5]],
        [Dvec[4], Dvec[5], Dvec[2]],
    ])
    D = (D + D.T) / 2

    eigenvalues, eigenvectors = np.linalg.eigh(D)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    fa = _fractional_anisotropy(eigenvalues)
    md = eigenvalues.mean()
    ad = eigenvalues[0]
    rd = eigenvalues[1:].mean()
    cl, cp, cs = _shape_indices(eigenvalues)
    mo = _mode_anisotropy(eigenvalues)
    vr = _volume_ratio(eigenvalues)

    return DiffusionTensor(
        tensor=D, eigenvalues=eigenvalues, eigenvectors=eigenvectors,
        fa=fa, md=md, ad=ad, rd=rd,
        cl=cl, cp=cp, cs=cs, mo=mo, vr=vr,
    )


@dataclass
class DiffusionKurtosis:
    mk: float = 0.0
    ak: float = 0.0
    rk: float = 0.0

    def to_dict(self) -> dict:
        return {
            "mk": round(self.mk, 6),
            "ak": round(self.ak, 6),
            "rk": round(self.rk, 6),
        }


def fit_dki(
    dw_signals: np.ndarray,
    b0: np.ndarray,
    gradients: np.ndarray,
    bvals: np.ndarray,
) -> DiffusionKurtosis:
    S = np.asarray(dw_signals, dtype=np.float64)
    S0 = np.asarray(b0, dtype=np.float64).mean()
    n_meas = S.shape[0]

    g = np.asarray(gradients, dtype=np.float64)
    gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]

    A = np.column_stack([
        -bvals * gx * gx,
        -bvals * gy * gy,
        -bvals * gz * gz,
        -2 * bvals * gx * gy,
        -2 * bvals * gx * gz,
        -2 * bvals * gy * gz,
        bvals ** 2 * gx ** 4 / 6,
        bvals ** 2 * gy ** 4 / 6,
        bvals ** 2 * gz ** 4 / 6,
        4 * bvals ** 2 * gx ** 2 * gy ** 2 / 6,
        4 * bvals ** 2 * gx ** 2 * gz ** 2 / 6,
        4 * bvals ** 2 * gy ** 2 * gz ** 2 / 6,
        2 * bvals ** 2 * gx ** 3 * gy / 6,
        2 * bvals ** 2 * gx ** 3 * gz / 6,
        2 * bvals ** 2 * gy ** 3 * gx / 6,
        2 * bvals ** 2 * gy ** 3 * gz / 6,
        2 * bvals ** 2 * gz ** 3 * gx / 6,
        2 * bvals ** 2 * gz ** 3 * gy / 6,
        4 * bvals ** 2 * gx ** 2 * gy * gz / 6,
        4 * bvals ** 2 * gy ** 2 * gx * gz / 6,
        4 * bvals ** 2 * gz ** 2 * gx * gy / 6,
    ])

    y = -np.log(np.maximum(S / (S0 + 1e-12), 1e-12))
    params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    Dvec = params[:6]
    Wvec = params[6:]

    D = np.array([
        [Dvec[0], Dvec[3], Dvec[4]],
        [Dvec[3], Dvec[1], Dvec[5]],
        [Dvec[4], Dvec[5], Dvec[2]],
    ])
    D = (D + D.T) / 2
    evals, evecs = np.linalg.eigh(D)
    evals = np.maximum(evals, 1e-12)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    md = evals.mean()
    ad = evals[0]
    rd = evals[1:].mean()

    W_rot = _rotate_kurtosis_tensor(Wvec, evecs)
    mk = (W_rot[0] * evals[0] ** 2 +
          W_rot[1] * evals[1] ** 2 +
          W_rot[2] * evals[2] ** 2 +
          2 * W_rot[3] * evals[0] * evals[1] +
          2 * W_rot[4] * evals[0] * evals[2] +
          2 * W_rot[5] * evals[1] * evals[2])
    mk = mk / (md ** 2) if md > 1e-12 else 0.0

    ak = W_rot[0] * evals[0] ** 2 / (ad ** 2) if ad > 1e-12 else 0.0
    rk = (W_rot[1] * evals[1] ** 2 + W_rot[2] * evals[2] ** 2) / (rd ** 2) if rd > 1e-12 else 0.0

    return DiffusionKurtosis(mk=float(mk), ak=float(ak), rk=float(rk))


def _rotate_kurtosis_tensor(Wvec: np.ndarray, evecs: np.ndarray) -> np.ndarray:
    W = np.zeros((6,))
    W[0] = Wvec[0]
    W[1] = Wvec[1]
    W[2] = Wvec[2]
    W[3] = Wvec[3]
    W[4] = Wvec[4]
    W[5] = Wvec[5]

    R = evecs
    W_rot = np.zeros(6)
    W_rot[0] = R[0, 0] ** 4 * W[0] + R[1, 0] ** 4 * W[1] + R[2, 0] ** 4 * W[2] \
             + 6 * R[0, 0] ** 2 * R[1, 0] ** 2 * W[3] + 6 * R[0, 0] ** 2 * R[2, 0] ** 2 * W[4] \
             + 6 * R[1, 0] ** 2 * R[2, 0] ** 2 * W[5]
    W_rot[1] = R[0, 1] ** 4 * W[0] + R[1, 1] ** 4 * W[1] + R[2, 1] ** 4 * W[2] \
             + 6 * R[0, 1] ** 2 * R[1, 1] ** 2 * W[3] + 6 * R[0, 1] ** 2 * R[2, 1] ** 2 * W[4] \
             + 6 * R[1, 1] ** 2 * R[2, 1] ** 2 * W[5]
    W_rot[2] = R[0, 2] ** 4 * W[0] + R[1, 2] ** 4 * W[1] + R[2, 2] ** 4 * W[2] \
             + 6 * R[0, 2] ** 2 * R[1, 2] ** 2 * W[3] + 6 * R[0, 2] ** 2 * R[2, 2] ** 2 * W[4] \
             + 6 * R[1, 2] ** 2 * R[2, 2] ** 2 * W[5]
    W_rot[3] = 2 * R[0, 0] ** 2 * R[0, 1] ** 2 * W[0] + 2 * R[1, 0] ** 2 * R[1, 1] ** 2 * W[1] \
             + 2 * R[2, 0] ** 2 * R[2, 1] ** 2 * W[2] \
             + (R[0, 0] ** 2 * R[1, 1] ** 2 + R[1, 0] ** 2 * R[0, 1] ** 2 + 4 * R[0, 0] * R[1, 0] * R[0, 1] * R[1, 1]) * W[3] \
             + (R[0, 0] ** 2 * R[2, 1] ** 2 + R[2, 0] ** 2 * R[0, 1] ** 2 + 4 * R[0, 0] * R[2, 0] * R[0, 1] * R[2, 1]) * W[4] \
             + (R[1, 0] ** 2 * R[2, 1] ** 2 + R[2, 0] ** 2 * R[1, 1] ** 2 + 4 * R[1, 0] * R[2, 0] * R[1, 1] * R[2, 1]) * W[5]
    W_rot[4] = 2 * R[0, 0] ** 2 * R[0, 2] ** 2 * W[0] + 2 * R[1, 0] ** 2 * R[1, 2] ** 2 * W[1] \
             + 2 * R[2, 0] ** 2 * R[2, 2] ** 2 * W[2] \
             + (R[0, 0] ** 2 * R[1, 2] ** 2 + R[1, 0] ** 2 * R[0, 2] ** 2 + 4 * R[0, 0] * R[1, 0] * R[0, 2] * R[1, 2]) * W[3] \
             + (R[0, 0] ** 2 * R[2, 2] ** 2 + R[2, 0] ** 2 * R[0, 2] ** 2 + 4 * R[0, 0] * R[2, 0] * R[0, 2] * R[2, 2]) * W[4] \
             + (R[1, 0] ** 2 * R[2, 2] ** 2 + R[2, 0] ** 2 * R[1, 2] ** 2 + 4 * R[1, 0] * R[2, 0] * R[1, 2] * R[2, 2]) * W[5]
    W_rot[5] = 2 * R[0, 1] ** 2 * R[0, 2] ** 2 * W[0] + 2 * R[1, 1] ** 2 * R[1, 2] ** 2 * W[1] \
             + 2 * R[2, 1] ** 2 * R[2, 2] ** 2 * W[2] \
             + (R[0, 1] ** 2 * R[1, 2] ** 2 + R[1, 1] ** 2 * R[0, 2] ** 2 + 4 * R[0, 1] * R[1, 1] * R[0, 2] * R[1, 2]) * W[3] \
             + (R[0, 1] ** 2 * R[2, 2] ** 2 + R[2, 1] ** 2 * R[0, 2] ** 2 + 4 * R[0, 1] * R[2, 1] * R[0, 2] * R[2, 2]) * W[4] \
             + (R[1, 1] ** 2 * R[2, 2] ** 2 + R[2, 1] ** 2 * R[1, 2] ** 2 + 4 * R[1, 1] * R[2, 1] * R[1, 2] * R[2, 2]) * W[5]
    return W_rot


def tensor_metrics(tensors: np.ndarray) -> dict:
    T = np.asarray(tensors, dtype=np.float64)
    evals = np.zeros((*T.shape[:-2], 3))
    evecs = np.zeros((*T.shape[:-2], 3, 3))
    fa_map = np.zeros(T.shape[:-2])
    md_map = np.zeros(T.shape[:-2])
    cl_map = np.zeros(T.shape[:-2])
    cp_map = np.zeros(T.shape[:-2])
    cs_map = np.zeros(T.shape[:-2])
    mo_map = np.zeros(T.shape[:-2])
    vr_map = np.zeros(T.shape[:-2])

    it = np.nditer(fa_map, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        D = T[idx]
        eigvals, eigvecs = np.linalg.eigh(D)
        eigvals = np.maximum(eigvals, 1e-12)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        evals[idx] = eigvals
        evecs[idx] = eigvecs
        fa_map[idx] = _fractional_anisotropy(eigvals)
        md_map[idx] = eigvals.mean()
        cl_map[idx], cp_map[idx], cs_map[idx] = _shape_indices(eigvals)
        mo_map[idx] = _mode_anisotropy(eigvals)
        vr_map[idx] = _volume_ratio(eigvals)

    color_fa = color_fa_map(fa_map, evecs)

    return {
        "fa_map": fa_map.tolist(),
        "md_map": md_map.tolist(),
        "cl_map": cl_map.tolist(),
        "cp_map": cp_map.tolist(),
        "cs_map": cs_map.tolist(),
        "mo_map": mo_map.tolist(),
        "vr_map": vr_map.tolist(),
        "color_fa": color_fa.tolist(),
        "eigenvalues": evals.tolist(),
        "eigenvectors": evecs.tolist(),
        "shape": list(T.shape[:-2]),
    }


def color_fa_map(fa: np.ndarray, evecs: np.ndarray) -> np.ndarray:
    primary = np.abs(evecs[..., :, 0])
    rgb = np.zeros((*fa.shape, 3))
    rgb[..., 0] = primary[..., 0] * fa
    rgb[..., 1] = primary[..., 1] * fa
    rgb[..., 2] = primary[..., 2] * fa
    max_rgb = rgb.max(axis=-1, keepdims=True)
    mask = max_rgb[..., 0] > 0
    rgb[mask] = rgb[mask] / max_rgb[mask]
    return rgb


def tensor_glyph(tensor: np.ndarray, n_points: int = 32) -> np.ndarray:
    D = np.asarray(tensor, dtype=np.float64)
    evals, evecs = np.linalg.eigh(D)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    evals = np.maximum(evals, 1e-12)

    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack([x, y, z], axis=-1)

    ellipsoid = sphere * np.sqrt(evals)
    glyph = ellipsoid @ evecs.T
    return glyph.tolist()
