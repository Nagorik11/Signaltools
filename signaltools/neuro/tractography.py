from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Streamline:
    points: np.ndarray  # Nx3
    length: float = 0.0
    mean_fa: float = 0.0

    def to_dict(self) -> dict:
        return {
            "points": self.points.tolist(),
            "length": round(self.length, 4),
            "n_points": len(self.points),
            "mean_fa": round(self.mean_fa, 4),
        }


@dataclass
class TractographyResult:
    streamlines: List[Streamline] = field(default_factory=list)
    n_streamlines: int = 0
    fiber_stats: dict = field(default_factory=dict)
    connectivity_matrix: List[List[float]] = field(default_factory=list)
    connectivity_labels: List[str] = field(default_factory=list)
    clusters: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_streamlines": len(self.streamlines),
            "streamlines": [s.to_dict() for s in self.streamlines],
            "fiber_stats": self.fiber_stats,
            "connectivity_matrix": self.connectivity_matrix,
            "connectivity_labels": self.connectivity_labels,
            "clusters": self.clusters,
        }


def track_streamlines(
    evals: np.ndarray,
    evecs: np.ndarray,
    fa_map: np.ndarray,
    seed_mask: np.ndarray,
    step_size: float = 0.5,
    min_fa: float = 0.15,
    max_angle: float = 45.0,
    max_steps: int = 500,
    n_seeds: int = 200,
    probabilistic: bool = True,
    n_regions: int = 6,
) -> TractographyResult:
    max_angle_rad = np.radians(max_angle)
    shape = fa_map.shape
    result = TractographyResult()

    seed_candidates = np.argwhere(seed_mask & (fa_map > min_fa))
    if len(seed_candidates) == 0:
        seed_candidates = np.argwhere(fa_map > min_fa)
    if len(seed_candidates) == 0:
        return result

    rng = np.random.default_rng(42)
    if n_seeds > 0 and len(seed_candidates) > n_seeds:
        idx = rng.choice(len(seed_candidates), n_seeds, replace=False)
        seed_candidates = seed_candidates[idx]

    streamlines = []
    for seed in seed_candidates:
        pos = seed.astype(np.float64) + 0.5
        fwd = _track_direction(pos, +1, evals, evecs, fa_map, shape,
                               step_size, min_fa, max_angle_rad, max_steps,
                               probabilistic, rng)
        bwd = _track_direction(pos, -1, evals, evecs, fa_map, shape,
                               step_size, min_fa, max_angle_rad, max_steps,
                               probabilistic, rng)

        if len(fwd) + len(bwd) > 5:
            bwd.reverse()
            full = bwd + fwd
            pts = np.array(full)
            # Compute mean FA along streamline
            mfa = _mean_fa_along(pts, fa_map)
            sl = Streamline(points=pts, length=_path_length(pts), mean_fa=mfa)
            streamlines.append(sl)

    result.streamlines = streamlines
    result.n_streamlines = len(streamlines)
    result.fiber_stats = _fiber_stats(streamlines)

    # Clustering
    clusters, _ = _cluster_streamlines(streamlines, n_clusters=min(5, len(streamlines)))
    result.clusters = clusters

    # Connectivity matrix
    labels = [f"Region_{i}" for i in range(n_regions)]
    result.connectivity_labels = labels
    cm = _connectivity_matrix(streamlines, shape, n_regions)
    result.connectivity_matrix = cm

    return result


def _track_direction(
    start: np.ndarray, direction: int,
    evals: np.ndarray, evecs: np.ndarray, fa: np.ndarray,
    shape: Tuple[int, ...], step: float, min_fa: float,
    max_angle: float, max_steps: int,
    probabilistic: bool, rng: np.random.Generator,
) -> List[np.ndarray]:
    pts = []
    pos = start.copy()

    eps = []
    for _ in range(max_steps + 1):
        idx = tuple(np.round(pos).astype(int))
        if not (0 <= idx[0] < shape[0] and 0 <= idx[1] < shape[1] and 0 <= idx[2] < shape[2]):
            break
        if fa[idx] < min_fa:
            break

        ev1 = np.array(evecs[idx][:, 0], dtype=np.float64)
        evals_vox = np.array(evals[idx], dtype=np.float64)

        if probabilistic:
            dir_vec = _sample_direction(ev1, evals_vox, fa[idx], rng)
        else:
            dir_vec = ev1

        # Align with previous direction
        if eps:
            if np.dot(dir_vec, eps[-1]) < 0:
                dir_vec = -dir_vec
            cos_angle = np.clip(np.dot(dir_vec, eps[-1]), -1, 1)
            if np.arccos(cos_angle) > max_angle:
                break
        else:
            dir_vec = direction * dir_vec

        pos = pos + dir_vec * step
        eps.append(dir_vec)
        pts.append(pos.copy())

    return pts


def _sample_direction(
    ev1: np.ndarray, evals: np.ndarray, fa_val: float,
    rng: np.random.Generator,
) -> np.ndarray:
    concentration = max(1, fa_val * 20)
    n = 3
    z = rng.normal(size=n)
    z = z / np.linalg.norm(z)
    scale = 1.0 / np.sqrt(concentration)
    perturb = ev1 + scale * z
    return perturb / (np.linalg.norm(perturb) + 1e-12)


def _mean_fa_along(pts: np.ndarray, fa_map: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    fas = []
    for p in pts:
        idx = tuple(np.round(p).astype(int))
        if 0 <= idx[0] < fa_map.shape[0] and 0 <= idx[1] < fa_map.shape[1] and 0 <= idx[2] < fa_map.shape[2]:
            fas.append(fa_map[idx])
    return float(np.mean(fas)) if fas else 0.0


def _fiber_stats(streamlines: List[Streamline]) -> dict:
    if not streamlines:
        return {"n": 0, "mean_length": 0, "std_length": 0, "mean_fa": 0, "lengths": []}
    lengths = [s.length for s in streamlines]
    mean_fas = [s.mean_fa for s in streamlines]
    return {
        "n": len(streamlines),
        "mean_length": round(float(np.mean(lengths)), 4),
        "std_length": round(float(np.std(lengths)), 4),
        "min_length": round(float(np.min(lengths)), 4),
        "max_length": round(float(np.max(lengths)), 4),
        "mean_fa": round(float(np.mean(mean_fas)), 4),
        "lengths": [round(float(l), 4) for l in lengths[:50]],
    }


def _cluster_streamlines(
    streamlines: List[Streamline], n_clusters: int = 5,
) -> Tuple[List[int], np.ndarray]:
    n = len(streamlines)
    if n < n_clusters or n < 3:
        return list(range(n)), np.array([[i] for i in range(n)])

    endpoints = []
    for sl in streamlines:
        pts = sl.points
        ep = np.concatenate([pts[0], pts[-1]])
        endpoints.append(ep)
    X = np.array(endpoints)

    centroids_idx = rng = np.random.default_rng(123).choice(n, n_clusters, replace=False)
    centroids = X[centroids_idx]
    labels = np.zeros(n, dtype=int)

    for _ in range(20):
        for i in range(n):
            dists = np.linalg.norm(X[i] - centroids, axis=1)
            labels[i] = int(np.argmin(dists))
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                new_centroids[k] = X[mask].mean(axis=0)
            else:
                new_centroids[k] = centroids[k]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels.tolist(), centroids


def _connectivity_matrix(
    streamlines: List[Streamline], shape: Tuple[int, ...], n_regions: int,
) -> List[List[float]]:
    Z, Y, X = shape
    cm = np.zeros((n_regions, n_regions))

    if len(streamlines) < 2:
        return cm.tolist()

    # Divide volume into regions
    region_map = np.zeros((Z, Y, X), dtype=int)
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                rz = min(z * n_regions // Z, n_regions - 1)
                ry = min(y * n_regions // Y, n_regions - 1)
                rx = min(x * n_regions // X, n_regions - 1)
                region_map[z, y, x] = (rz * n_regions + ry) % n_regions

    for sl in streamlines:
        pts = sl.points
        if len(pts) < 2:
            continue
        start_idx = tuple(np.round(pts[0]).astype(int))
        end_idx = tuple(np.round(pts[-1]).astype(int))
        try:
            r1 = region_map[start_idx]
            r2 = region_map[end_idx]
            if r1 != r2:
                cm[r1, r2] += 1
                cm[r2, r1] += 1
        except IndexError:
            pass

    # Normalize
    max_cm = cm.max()
    if max_cm > 0:
        cm = cm / max_cm

    return cm.tolist()


def _path_length(pts: np.ndarray) -> float:
    return float(np.sum(np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))))


def bundle_centroid(streamlines: List[Streamline]) -> Optional[np.ndarray]:
    if not streamlines:
        return None
    max_len = max(len(s.points) for s in streamlines)
    resampled = []
    for sl in streamlines:
        pts = sl.points
        if len(pts) < 2:
            continue
        indices = np.linspace(0, len(pts) - 1, max_len)
        x = np.interp(indices, np.arange(len(pts)), pts[:, 0])
        y = np.interp(indices, np.arange(len(pts)), pts[:, 1])
        z = np.interp(indices, np.arange(len(pts)), pts[:, 2])
        resampled.append(np.stack([x, y, z], axis=1))
    if not resampled:
        return None
    return np.mean(resampled, axis=0)
