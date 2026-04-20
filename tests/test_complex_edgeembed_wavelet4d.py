from __future__ import annotations

import numpy as np

import signaltools as st


def test_analytic_multichannel_and_complex_mix() -> None:
    features = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, -1.0],
        [-1.0, 0.0],
    ], dtype=float)
    analytic = st.analytic_signal_multichannel(features)
    assert np.asarray(analytic.real).shape == features.shape
    assert np.asarray(analytic.imag).shape == features.shape

    complex_features = np.asarray(analytic.real) + 1j * np.asarray(analytic.imag)
    mixed = st.complex_channel_mix(complex_features, out_channels=3)
    assert np.asarray(mixed.real).shape == (4, 3)



def test_structured_edge_embedding_attention_and_stack() -> None:
    adjacency = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=float,
    )
    features = np.array(
        [[1.0, 0.1, -0.2], [0.5, -0.5, 0.3], [2.0, 1.0, -1.0], [-1.0, 0.2, 0.8]],
        dtype=float,
    )
    edge_embeddings = np.ones((4, 4, 3), dtype=float) * 0.05

    attn = st.structured_edge_embedding_attention(features, adjacency, edge_embeddings, num_heads=2)
    assert np.asarray(attn.output).shape == features.shape
    assert attn.meta["structured_edge_embeddings"] is True

    stack = st.graph_transformer_edge_embedding_stack(features, adjacency, edge_embeddings, depth=2, num_heads=2)
    assert np.asarray(stack.output).shape == features.shape
    assert len(stack.layers) == 2



def test_wavelet_packet_4d_tensor() -> None:
    tensor = np.arange(2 * 4 * 4 * 4, dtype=float).reshape(2, 4, 4, 4)
    tree = st.wavelet_packet_4d_decompose(tensor, level=1, family="haar")
    recon = st.wavelet_packet_4d_reconstruct(tree)
    recon_arr = np.asarray(recon, dtype=float)
    assert recon_arr.ndim == 4
    assert all(dim > 0 for dim in recon_arr.shape)
