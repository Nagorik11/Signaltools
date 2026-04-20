from __future__ import annotations

import numpy as np
import pytest

import signaltools as st


def test_adaptive_wiener() -> None:
    x = [0.0, 1.0, 0.8, 1.2, 1.0, 0.9]
    result = st.adaptive_wiener_filter_1d(x, window_size=3, adaptation_rate=0.2)
    assert len(result.filtered) == len(x)
    assert len(result.noise_trace) == len(x)
    assert len(result.local_variance) == len(x)


def test_ekf_and_ukf() -> None:
    z = [0.1, 0.2, 0.15, 0.25]
    f = lambda x: np.array([x[0]])
    h = lambda x: np.array([x[0] ** 2])
    F = lambda x: np.array([[1.0]])
    H = lambda x: np.array([[2.0 * x[0]]])
    ekf = st.extended_kalman_filter(
        z,
        transition_fn=f,
        measurement_fn=h,
        transition_jacobian=F,
        measurement_jacobian=H,
        initial_state=[0.3],
        initial_covariance=[[1.0]],
        process_covariance=[[1e-4]],
        measurement_covariance=[[1e-2]],
    )
    ukf = st.unscented_kalman_filter(
        z,
        transition_fn=f,
        measurement_fn=h,
        initial_state=[0.3],
        initial_covariance=[[1.0]],
        process_covariance=[[1e-4]],
        measurement_covariance=[[1e-2]],
    )
    assert len(ekf.estimates) == len(z)
    assert len(ukf.estimates) == len(z)


def test_particle_filter() -> None:
    z = [0.1, 0.15, 0.2, 0.18]
    pf = st.particle_filter_1d(z, num_particles=32, process_std=0.05, measurement_std=0.1)
    assert len(pf.estimates) == len(z)
    assert len(pf.particles) == len(z)
    assert len(pf.weights) == len(z)


def test_image_morphology_2d() -> None:
    image = np.array([[0, 0, 1], [0, 2, 0], [1, 0, 0]], dtype=float)
    assert np.asarray(st.dilation_2d(image, 3)).shape == image.shape
    assert np.asarray(st.erosion_2d(image, 3)).shape == image.shape
    assert np.asarray(st.opening_2d(image, 3)).shape == image.shape
    assert np.asarray(st.closing_2d(image, 3)).shape == image.shape
    assert np.asarray(st.median_filter_2d(image, 3)).shape == image.shape
    assert np.asarray(st.morphological_gradient_2d(image, 3)).shape == image.shape
    with pytest.raises(ValueError):
        st.dilation_2d([1, 2, 3], 3)


def test_filter_banks() -> None:
    x = np.sin(2 * np.pi * 0.05 * np.arange(64)).tolist()
    haar = st.haar_analysis_bank(x)
    recon = st.haar_synthesis_bank(haar.subbands[0], haar.subbands[1])
    uniform = st.uniform_filter_bank(x, bands=4, fir_taps=15)
    uniform_recon = st.reconstruct_uniform_filter_bank(uniform.subbands, bands=4, fir_taps=15)
    assert len(haar.subbands) == 2
    assert len(recon) > 0
    assert len(uniform.subbands) == 4
    assert len(uniform_recon) > 0


def test_graph_wavelets_and_chebyshev() -> None:
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    signal = [1.0, 2.0, 3.0]
    cheb = st.chebyshev_graph_filter(signal, adjacency, coeffs=[1.0, -0.25, 0.1])
    wavelets = st.graph_wavelet_transform(signal, adjacency, scales=[0.5, 1.0], kind="heat")
    kernel = st.graph_wavelet_kernel(np.array([0.0, 1.0, 2.0]), scale=1.0, kind="mexican_hat")
    assert len(cheb) == 3
    assert len(wavelets) == 2
    assert kernel.shape == (3,)
    with pytest.raises(ValueError):
        st.graph_wavelet_kernel(np.array([0.0, 1.0]), scale=1.0, kind="weird")
