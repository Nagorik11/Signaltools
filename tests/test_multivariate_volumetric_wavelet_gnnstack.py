from __future__ import annotations

import numpy as np
import pytest

import signaltools as st


def test_particle_filter_multivariate() -> None:
    z = np.array([[0.1, 0.2], [0.2, 0.3], [0.15, 0.25]], dtype=float)
    transition = lambda particles, rng: particles + rng.normal(0.0, 0.05, size=particles.shape)
    likelihood = lambda measurement, particles: np.exp(-0.5 * np.sum((particles - measurement) ** 2, axis=1) / 0.1**2)
    init = np.zeros((16, 2), dtype=float)
    result = st.particle_filter_multivariate(z, transition, likelihood, init)
    assert len(result.estimates) == len(z)
    with pytest.raises(ValueError):
        st.particle_filter_multivariate([1, 2, 3], transition, likelihood, init)


def test_volumetric_arbitrary_kernels() -> None:
    vol = np.zeros((3, 3, 3), dtype=float)
    vol[1, 1, 1] = 1.0
    kernel = np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]]], dtype=float)
    assert np.asarray(st.dilation_3d_kernel(vol, kernel)).shape == vol.shape
    assert np.asarray(st.erosion_3d_kernel(vol, kernel)).shape == vol.shape
    assert np.asarray(st.opening_3d_kernel(vol, kernel)).shape == vol.shape
    assert np.asarray(st.closing_3d_kernel(vol, kernel)).shape == vol.shape


def test_wavelet_packet_families() -> None:
    x = np.sin(2 * np.pi * 0.05 * np.arange(32)).tolist()
    for family in ['haar', 'db2', 'sym2']:
        tree = st.wavelet_packet_decompose(x, level=2, family=family)
        recon = st.wavelet_packet_reconstruct(tree)
        assert len(recon) > 0
    with pytest.raises(ValueError):
        st.wavelet_filters('weird')


def test_gnn_stack_and_pooling() -> None:
    adjacency = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=float)
    signal = [1.0, -1.0, 2.0, -2.0]
    result = st.stacked_gnn(signal, adjacency, layer_weights=[[1.0, -0.25], [1.0, 0.1]], activation='relu', pooling_factor=2, pooling_mode='mean', residual=True)
    assert len(result.layers) == 2
    assert len(result.pooled) == 2
    assert len(st.graph_pool(signal, factor=2, mode='max')) == 2
    with pytest.raises(ValueError):
        st.graph_pool(signal, factor=2, mode='weird')
