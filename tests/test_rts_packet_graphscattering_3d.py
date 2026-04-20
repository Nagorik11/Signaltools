from __future__ import annotations

import numpy as np
import pytest

import signaltools as st


def test_backward_and_rts_smoothers() -> None:
    x = [1.0, 0.5, 0.25, 0.1]
    backward = st.backward_exponential_smoother(x, alpha=0.5)
    filtered_estimates = [[1.0], [0.8], [0.6]]
    filtered_covariances = [[[1.0]], [[0.8]], [[0.6]]]
    smoothed = st.rts_smoother(filtered_estimates, filtered_covariances, [[1.0]], [[0.01]])
    assert len(backward) == len(x)
    assert len(smoothed.estimates) == 3
    with pytest.raises(ValueError):
        st.rts_smoother([1, 2, 3], [[[1.0]]], [[1.0]], [[0.01]])


def test_nonlinear_particle_filter() -> None:
    z = [0.1, 0.2, 0.15]
    transition = lambda particles, rng: particles + rng.normal(0.0, 0.05, size=len(particles))
    likelihood = lambda measurement, particles: np.exp(-0.5 * ((measurement - particles) / 0.1) ** 2)
    result = st.particle_filter_nonlinear(z, transition, likelihood, initial_particles=np.linspace(-1, 1, 32))
    assert len(result.estimates) == len(z)
    with pytest.raises(ValueError):
        st.particle_filter_nonlinear(z, transition, likelihood, initial_particles=np.zeros((2, 2)))


def test_morphology_3d() -> None:
    volume = np.zeros((3, 3, 3), dtype=float)
    volume[1, 1, 1] = 1.0
    assert np.asarray(st.dilation_3d(volume, 3)).shape == volume.shape
    assert np.asarray(st.erosion_3d(volume, 3)).shape == volume.shape
    assert np.asarray(st.opening_3d(volume, 3)).shape == volume.shape
    assert np.asarray(st.closing_3d(volume, 3)).shape == volume.shape
    assert np.asarray(st.median_filter_3d(volume, 3)).shape == volume.shape
    assert np.asarray(st.morphological_gradient_3d(volume, 3)).shape == volume.shape


def test_wavelet_packet() -> None:
    x = np.sin(2 * np.pi * 0.05 * np.arange(32)).tolist()
    tree = st.wavelet_packet_decompose(x, level=2)
    recon = st.wavelet_packet_reconstruct(tree)
    assert '' in tree.nodes
    assert len(recon) > 0


def test_graph_scattering_and_spectral_gnn() -> None:
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    signal = [1.0, -2.0, 3.0]
    gnn = st.spectral_gnn_filter(signal, adjacency, weights=[1.0, -0.25, 0.1], activation='tanh')
    scat = st.graph_scattering_transform(signal, adjacency, scales=[0.5, 1.0])
    assert len(gnn) == 3
    assert len(scat) == 2
    with pytest.raises(ValueError):
        st.spectral_gnn_filter(signal, adjacency, weights=[1.0], activation='weird')
