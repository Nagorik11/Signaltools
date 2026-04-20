from __future__ import annotations

import numpy as np
import pytest

import signaltools as st


def test_kalman_and_wiener() -> None:
    x = [0.0, 1.0, 0.8, 1.2, 1.0, 0.9]
    kalman = st.kalman_filter_1d(x, process_variance=1e-4, measurement_variance=1e-2)
    wiener = st.wiener_filter_1d(x, window_size=3)
    assert len(kalman.estimates) == len(x)
    assert len(kalman.gains) == len(x)
    assert len(wiener.filtered) == len(x)
    assert wiener.noise_variance >= 0
    assert st.kalman_filter_1d([], process_variance=1e-4, measurement_variance=1e-2).estimates == []


def test_morphological_filters() -> None:
    x = [0, 1, 5, 1, 0, 3, 0]
    assert len(st.advanced_median_filter(x, 3)) == len(x)
    assert len(st.rank_filter(x, 3, rank=0)) == len(x)
    assert len(st.dilation_1d(x, 3)) == len(x)
    assert len(st.erosion_1d(x, 3)) == len(x)
    assert len(st.opening_1d(x, 3)) == len(x)
    assert len(st.closing_1d(x, 3)) == len(x)
    assert len(st.morphological_gradient_1d(x, 3)) == len(x)
    with pytest.raises(ValueError):
        st.rank_filter(x, 4, rank=0)
    with pytest.raises(ValueError):
        st.rank_filter(x, 3, rank=3)


def test_graph_filters() -> None:
    adjacency = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=float)
    signal = [1.0, 2.0, 3.0]
    L = st.graph_laplacian(adjacency)
    eigvals, eigvecs = st.graph_fourier_basis(adjacency)
    filtered = st.graph_filter_signal(signal, adjacency)
    poly = st.graph_polynomial_filter(signal, adjacency, coeffs=[1.0, -0.25])
    assert L.shape == (3, 3)
    assert eigvals.shape == (3,)
    assert eigvecs.shape == (3, 3)
    assert len(filtered) == 3
    assert len(poly) == 3
    with pytest.raises(ValueError):
        st.graph_laplacian([[1, 2, 3]], normalized=True)
    with pytest.raises(ValueError):
        st.graph_filter_signal([1.0, 2.0], adjacency)


def test_multirate_polyphase() -> None:
    x = np.sin(2 * np.pi * 0.05 * np.arange(64)).tolist()
    coeffs = [1, 2, 3, 4, 5, 6]
    phases = st.polyphase_decompose(coeffs, 2)
    dec = st.decimate(x, 2, fir_taps=15)
    interp = st.interpolate(x, 2, fir_taps=15)
    low, high = st.two_band_analysis_bank(x, fir_taps=15)
    assert len(phases) == 2
    assert sum(len(p) for p in phases) == len(coeffs)
    assert len(dec) > 0
    assert len(interp) == len(x) * 2
    assert len(low) > 0 and len(high) > 0
    assert st.decimate([], 2) == []
    assert st.interpolate([], 2) == []
