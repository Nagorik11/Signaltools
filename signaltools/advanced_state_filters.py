"""Advanced nonlinear and Bayesian filtering routines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from .utils import ensure_non_negative_float, ensure_positive_int, to_1d_float_array


@dataclass
class NonlinearFilterResult:
    estimates: list[float] | list[list[float]]
    covariances: list[float] | list[list[list[float]]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParticleFilterResult:
    estimates: list[float]
    particles: list[list[float]]
    weights: list[list[float]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveWienerResult:
    filtered: list[float]
    noise_trace: list[float]
    local_variance: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def adaptive_wiener_filter_1d(
    signal: list[float] | list[int],
    window_size: int = 5,
    adaptation_rate: float = 0.1,
    pad_mode: str = "reflect",
) -> AdaptiveWienerResult:
    x = to_1d_float_array(signal)
    window_size = ensure_positive_int(window_size, "window_size")
    adaptation_rate = ensure_non_negative_float(adaptation_rate, "adaptation_rate")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if x.size == 0:
        return AdaptiveWienerResult(filtered=[], noise_trace=[], local_variance=[])

    half = window_size // 2
    padded = np.pad(x, (half, half), mode=pad_mode)
    y = np.zeros_like(x)
    local_var = np.zeros_like(x)
    noise_trace = np.zeros_like(x)
    noise_est = float(np.var(x))

    for i in range(len(x)):
        window = padded[i : i + window_size]
        mu = float(np.mean(window))
        sigma2 = float(np.var(window))
        local_var[i] = sigma2
        noise_est = (1.0 - adaptation_rate) * noise_est + adaptation_rate * sigma2
        noise_trace[i] = noise_est
        gain = max(sigma2 - noise_est, 0.0) / max(sigma2, 1e-12)
        y[i] = mu + gain * (x[i] - mu)

    return AdaptiveWienerResult(filtered=y.tolist(), noise_trace=noise_trace.tolist(), local_variance=local_var.tolist())


def extended_kalman_filter(
    measurements: list[float] | list[int],
    transition_fn: Callable[[np.ndarray], np.ndarray],
    measurement_fn: Callable[[np.ndarray], np.ndarray],
    transition_jacobian: Callable[[np.ndarray], np.ndarray],
    measurement_jacobian: Callable[[np.ndarray], np.ndarray],
    initial_state: list[float] | np.ndarray,
    initial_covariance: list[list[float]] | np.ndarray,
    process_covariance: list[list[float]] | np.ndarray,
    measurement_covariance: list[list[float]] | np.ndarray,
) -> NonlinearFilterResult:
    z = to_1d_float_array(measurements, name="measurements")
    x = np.asarray(initial_state, dtype=np.float64).reshape(-1, 1)
    P = np.asarray(initial_covariance, dtype=np.float64)
    Q = np.asarray(process_covariance, dtype=np.float64)
    R = np.asarray(measurement_covariance, dtype=np.float64)

    estimates: list[list[float]] = []
    covariances: list[list[list[float]]] = []

    for measurement in z:
        x_pred = np.asarray(transition_fn(x.ravel()), dtype=np.float64).reshape(-1, 1)
        F = np.asarray(transition_jacobian(x.ravel()), dtype=np.float64)
        P_pred = F @ P @ F.T + Q

        H = np.asarray(measurement_jacobian(x_pred.ravel()), dtype=np.float64)
        y = np.array([[measurement]], dtype=np.float64) - np.asarray(measurement_fn(x_pred.ravel()), dtype=np.float64).reshape(-1, 1)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ y
        P = (np.eye(P.shape[0]) - K @ H) @ P_pred

        estimates.append(x.ravel().tolist())
        covariances.append(P.tolist())

    return NonlinearFilterResult(estimates=estimates, covariances=covariances, meta={"type": "EKF"})


def _sigma_points(x: np.ndarray, P: np.ndarray, alpha: float, beta: float, kappa: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.size
    lam = alpha**2 * (n + kappa) - n
    sigma = np.zeros((2 * n + 1, n), dtype=np.float64)
    sigma[0] = x
    sqrt = np.linalg.cholesky((n + lam) * P)
    for i in range(n):
        sigma[i + 1] = x + sqrt[:, i]
        sigma[n + i + 1] = x - sqrt[:, i]
    wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)), dtype=np.float64)
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = wm[0] + (1 - alpha**2 + beta)
    return sigma, wm, wc


def unscented_kalman_filter(
    measurements: list[float] | list[int],
    transition_fn: Callable[[np.ndarray], np.ndarray],
    measurement_fn: Callable[[np.ndarray], np.ndarray],
    initial_state: list[float] | np.ndarray,
    initial_covariance: list[list[float]] | np.ndarray,
    process_covariance: list[list[float]] | np.ndarray,
    measurement_covariance: list[list[float]] | np.ndarray,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> NonlinearFilterResult:
    z = to_1d_float_array(measurements, name="measurements")
    x = np.asarray(initial_state, dtype=np.float64).reshape(-1)
    P = np.asarray(initial_covariance, dtype=np.float64)
    Q = np.asarray(process_covariance, dtype=np.float64)
    R = np.asarray(measurement_covariance, dtype=np.float64)

    estimates: list[list[float]] = []
    covariances: list[list[list[float]]] = []

    for measurement in z:
        sigma, wm, wc = _sigma_points(x, P, alpha, beta, kappa)
        sigma_pred = np.array([transition_fn(point) for point in sigma], dtype=np.float64)
        x_pred = np.sum(wm[:, None] * sigma_pred, axis=0)
        P_pred = Q.copy()
        for i in range(len(sigma_pred)):
            diff = sigma_pred[i] - x_pred
            P_pred += wc[i] * np.outer(diff, diff)

        z_sigma = np.array([measurement_fn(point) for point in sigma_pred], dtype=np.float64).reshape(len(sigma_pred), -1)
        z_pred = np.sum(wm[:, None] * z_sigma, axis=0)
        S = R.copy()
        Cxz = np.zeros((x_pred.size, z_pred.size), dtype=np.float64)
        for i in range(len(sigma_pred)):
            dx = sigma_pred[i] - x_pred
            dz = z_sigma[i] - z_pred
            S += wc[i] * np.outer(dz, dz)
            Cxz += wc[i] * np.outer(dx, dz)
        K = Cxz @ np.linalg.inv(S)
        innovation = np.array([measurement], dtype=np.float64) - z_pred
        x = x_pred + (K @ innovation).ravel()
        P = P_pred - K @ S @ K.T

        estimates.append(x.tolist())
        covariances.append(P.tolist())

    return NonlinearFilterResult(estimates=estimates, covariances=covariances, meta={"type": "UKF", "alpha": alpha, "beta": beta, "kappa": kappa})


def particle_filter_1d(
    measurements: list[float] | list[int],
    num_particles: int = 100,
    process_std: float = 0.1,
    measurement_std: float = 0.2,
    initial_particles: list[float] | None = None,
    seed: int = 42,
) -> ParticleFilterResult:
    z = to_1d_float_array(measurements, name="measurements")
    num_particles = ensure_positive_int(num_particles, "num_particles")
    process_std = ensure_non_negative_float(process_std, "process_std")
    measurement_std = ensure_non_negative_float(measurement_std, "measurement_std")

    rng = np.random.default_rng(seed)
    if z.size == 0:
        return ParticleFilterResult(estimates=[], particles=[], weights=[], meta={"num_particles": num_particles})
    particles = np.asarray(initial_particles if initial_particles is not None else rng.normal(loc=float(z[0]), scale=max(measurement_std, 1e-6), size=num_particles), dtype=np.float64)
    weights = np.full(num_particles, 1.0 / num_particles, dtype=np.float64)

    estimates: list[float] = []
    particle_trace: list[list[float]] = []
    weight_trace: list[list[float]] = []

    for measurement in z:
        particles = particles + rng.normal(0.0, process_std, size=num_particles)
        likelihood = np.exp(-0.5 * ((measurement - particles) / max(measurement_std, 1e-12)) ** 2)
        likelihood /= max(measurement_std * np.sqrt(2.0 * np.pi), 1e-12)
        weights *= likelihood
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            weights[:] = 1.0 / num_particles
        else:
            weights /= weights_sum
        estimate = float(np.sum(weights * particles))
        estimates.append(estimate)
        particle_trace.append(particles.tolist())
        weight_trace.append(weights.tolist())

        indices = rng.choice(np.arange(num_particles), size=num_particles, p=weights)
        particles = particles[indices]
        weights[:] = 1.0 / num_particles

    return ParticleFilterResult(estimates=estimates, particles=particle_trace, weights=weight_trace, meta={"num_particles": num_particles, "process_std": process_std, "measurement_std": measurement_std})




@dataclass
class SmootherResult:
    estimates: list[float] | list[list[float]]
    covariances: list[float] | list[list[list[float]]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def backward_exponential_smoother(signal: list[float] | list[int], alpha: float = 0.3) -> list[float]:
    """Simple backward smoother over a 1D sequence."""
    x = to_1d_float_array(signal)
    alpha = ensure_non_negative_float(alpha, "alpha")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if x.size == 0:
        return []
    y = np.array(x, copy=True)
    for i in range(len(x) - 2, -1, -1):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i + 1]
    return y.tolist()


def rts_smoother(
    filtered_estimates: list[list[float]] | np.ndarray,
    filtered_covariances: list[list[list[float]]] | np.ndarray,
    transition_matrix: list[list[float]] | np.ndarray,
    process_covariance: list[list[float]] | np.ndarray,
) -> SmootherResult:
    """Rauch-Tung-Striebel backward smoother."""
    Xf = np.asarray(filtered_estimates, dtype=np.float64)
    Pf = np.asarray(filtered_covariances, dtype=np.float64)
    F = np.asarray(transition_matrix, dtype=np.float64)
    Q = np.asarray(process_covariance, dtype=np.float64)
    if Xf.ndim != 2 or Pf.ndim != 3:
        raise ValueError("filtered_estimates must be 2D and filtered_covariances must be 3D")
    n = Xf.shape[0]
    Xs = np.array(Xf, copy=True)
    Ps = np.array(Pf, copy=True)
    for k in range(n - 2, -1, -1):
        P_pred = F @ Pf[k] @ F.T + Q
        Ck = Pf[k] @ F.T @ np.linalg.inv(P_pred)
        Xs[k] = Xf[k] + Ck @ (Xs[k + 1] - F @ Xf[k])
        Ps[k] = Pf[k] + Ck @ (Ps[k + 1] - P_pred) @ Ck.T
    return SmootherResult(estimates=Xs.tolist(), covariances=Ps.tolist(), meta={"type": "RTS"})


def particle_filter_nonlinear(
    measurements: list[float] | list[int],
    transition_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    likelihood_fn: Callable[[float, np.ndarray], np.ndarray],
    initial_particles: list[float] | np.ndarray,
    num_particles: int | None = None,
    seed: int = 42,
) -> ParticleFilterResult:
    """Particle filter with user-supplied nonlinear dynamics and likelihood."""
    z = to_1d_float_array(measurements, name="measurements")
    particles = np.asarray(initial_particles, dtype=np.float64)
    if particles.ndim != 1:
        raise ValueError("initial_particles must be 1D")
    if num_particles is None:
        num_particles = len(particles)
    num_particles = ensure_positive_int(int(num_particles), "num_particles")
    if len(particles) != num_particles:
        raise ValueError("num_particles must match the length of initial_particles")
    rng = np.random.default_rng(seed)
    weights = np.full(num_particles, 1.0 / num_particles, dtype=np.float64)
    estimates: list[float] = []
    particle_trace: list[list[float]] = []
    weight_trace: list[list[float]] = []
    for measurement in z:
        particles = np.asarray(transition_fn(particles, rng), dtype=np.float64)
        likelihood = np.asarray(likelihood_fn(float(measurement), particles), dtype=np.float64)
        weights *= np.maximum(likelihood, 1e-300)
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            weights[:] = 1.0 / num_particles
        else:
            weights /= weights_sum
        estimates.append(float(np.sum(weights * particles)))
        particle_trace.append(particles.tolist())
        weight_trace.append(weights.tolist())
        idx = rng.choice(np.arange(num_particles), size=num_particles, p=weights)
        particles = particles[idx]
        weights[:] = 1.0 / num_particles
    return ParticleFilterResult(estimates=estimates, particles=particle_trace, weights=weight_trace, meta={"type": "nonlinear_particle", "num_particles": num_particles})



def particle_filter_multivariate(
    measurements: list[list[float]] | np.ndarray,
    transition_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    likelihood_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    initial_particles: list[list[float]] | np.ndarray,
    seed: int = 42,
) -> ParticleFilterResult:
    """Multivariate particle filter with configurable nonlinear dynamics."""
    Z = np.asarray(measurements, dtype=np.float64)
    particles = np.asarray(initial_particles, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("measurements must be a 2D array-like")
    if particles.ndim != 2:
        raise ValueError("initial_particles must be a 2D array-like")
    num_particles = particles.shape[0]
    rng = np.random.default_rng(seed)
    weights = np.full(num_particles, 1.0 / num_particles, dtype=np.float64)
    estimates: list[list[float]] = []
    particle_trace: list[list[float]] = []
    weight_trace: list[list[float]] = []
    for measurement in Z:
        particles = np.asarray(transition_fn(particles, rng), dtype=np.float64)
        likelihood = np.asarray(likelihood_fn(np.asarray(measurement, dtype=np.float64), particles), dtype=np.float64)
        if likelihood.ndim != 1 or len(likelihood) != num_particles:
            raise ValueError("likelihood_fn must return a 1D likelihood per particle")
        weights *= np.maximum(likelihood, 1e-300)
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            weights[:] = 1.0 / num_particles
        else:
            weights /= weights_sum
        estimates.append(np.sum(weights[:, None] * particles, axis=0).tolist())
        particle_trace.append(particles.tolist())
        weight_trace.append(weights.tolist())
        idx = rng.choice(np.arange(num_particles), size=num_particles, p=weights)
        particles = particles[idx]
        weights[:] = 1.0 / num_particles
    return ParticleFilterResult(estimates=estimates, particles=particle_trace, weights=weight_trace, meta={"type": "multivariate_particle", "num_particles": num_particles, "state_dim": particles.shape[1]})


__all__ = [
    "AdaptiveWienerResult",
    "NonlinearFilterResult",
    "ParticleFilterResult",
    "SmootherResult",
    "adaptive_wiener_filter_1d",
    "extended_kalman_filter",
    "unscented_kalman_filter",
    "particle_filter_1d",
    "backward_exponential_smoother",
    "rts_smoother",
    "particle_filter_nonlinear",
    "particle_filter_multivariate",
]
