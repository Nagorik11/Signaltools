from __future__ import annotations

import numpy as np
import pytest

import signaltools as st


def test_extended_wavelet_families_are_available() -> None:
    families = st.available_wavelet_families()
    for name in ["haar", "db4", "db6", "coif1", "coif2", "sym4"]:
        assert name in families
        h0, h1, g0, g1 = st.wavelet_filters(name)
        assert len(h0) == len(h1) == len(g0) == len(g1)


def test_extended_wavelet_packet_decomposition_reconstruction() -> None:
    x = np.sin(2 * np.pi * 0.05 * np.arange(64)).tolist()
    for family in ["db4", "db6", "coif1", "coif2", "sym4"]:
        tree = st.wavelet_packet_decompose(x, level=2, family=family)
        recon = st.wavelet_packet_reconstruct(tree)
        assert len(recon) >= 32
        assert tree.meta["wavelet"] == family


def test_graph_attention_matrix_rows_sum_to_one() -> None:
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    signal = [1.0, 2.0, -1.0]
    attn = np.asarray(st.graph_attention_matrix(signal, adjacency, alpha=0.7), dtype=float)
    assert attn.shape == adjacency.shape
    assert np.allclose(attn.sum(axis=1), 1.0)


def test_graph_attention_filter_and_deep_stack() -> None:
    adjacency = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=float,
    )
    signal = [1.0, -1.0, 2.0, -2.0]
    filtered = st.graph_attention_filter(signal, adjacency, alpha=0.5)
    assert len(filtered) == len(signal)

    result = st.deep_gnn_stack(
        signal,
        adjacency,
        layer_weights=[[1.0, -0.25], [0.9, 0.1, -0.05], [1.0]],
        activation="gelu",
        normalization="layernorm",
        attention=True,
        attention_alpha=0.8,
        attention_mix=0.35,
        pooling_factor=2,
        pooling_mode="mean",
        residual=True,
    )
    assert len(result.layers) == 3
    assert len(result.attention_matrices) == 3
    assert len(result.pooled) == 3
    assert len(result.output) == len(signal)
    assert result.meta["depth"] == 3


def test_deep_gnn_invalid_options_raise() -> None:
    adjacency = np.array([[0, 1], [1, 0]], dtype=float)
    signal = [1.0, 2.0]
    with pytest.raises(ValueError):
        st.deep_gnn_stack(signal, adjacency, [[1.0]], normalization="bad")
    with pytest.raises(ValueError):
        st.deep_gnn_stack(signal, adjacency, [[1.0]], attention=True, attention_mix=1.5)
    with pytest.raises(ValueError):
        st.graph_attention_matrix([1.0], adjacency)
