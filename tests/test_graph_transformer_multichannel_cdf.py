from __future__ import annotations

import numpy as np

import signaltools as st


def test_cdf_and_spline_families_available() -> None:
    families = st.available_wavelet_families()
    for family in ["cdf53", "cdf97", "spline53", "spline97", "bior22", "bior44"]:
        assert family in families
        assert st.wavelet_family_kind(family) == "biorthogonal"



def test_wavelet_packet_with_extended_biorthogonal_families() -> None:
    x = np.linspace(0.0, 1.0, 64).tolist()
    for family in ["cdf53", "cdf97", "spline53", "spline97", "bior22", "bior44"]:
        tree = st.wavelet_packet_decompose(x, level=2, family=family)
        recon = st.wavelet_packet_reconstruct(tree)
        assert len(recon) > 0
        assert tree.meta["kind"] == "biorthogonal"



def test_multichannel_attention_and_spectral_filter() -> None:
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    features = np.array([[1.0, 0.0], [0.5, -0.5], [2.0, 1.0]], dtype=float)

    attn = st.multihead_graph_attention_multichannel(features, adjacency, num_heads=3, alpha=0.7)
    assert np.asarray(attn.output).shape == features.shape
    assert len(attn.heads) == 3

    filtered = st.spectral_gnn_filter_multichannel(features, adjacency, weights=[1.0, -0.2, 0.05])
    assert np.asarray(filtered).shape == features.shape

    normed = st.graph_block_normalize_multichannel(features, mode="batchnorm")
    assert np.asarray(normed).shape == features.shape



def test_graph_transformer_layer_and_stack() -> None:
    adjacency = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=float,
    )
    features = np.array(
        [[1.0, 0.1, -0.2], [0.5, -0.5, 0.3], [2.0, 1.0, -1.0], [-1.0, 0.2, 0.8]],
        dtype=float,
    )

    layer = st.graph_transformer_layer(features, adjacency, num_heads=2, normalization="layernorm")
    assert np.asarray(layer.output).shape == features.shape
    assert layer.meta["channels"] == 3

    stack = st.graph_transformer_stack(features, adjacency, depth=3, num_heads=2, normalization="batchnorm")
    assert np.asarray(stack.output).shape == features.shape
    assert len(stack.layers) == 3
    assert stack.meta["depth"] == 3
