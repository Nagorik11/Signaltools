from __future__ import annotations

import numpy as np

import signaltools as st


def test_edge_message_passing_and_stack() -> None:
    adjacency = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
    node_features = np.array([[1.0, 0.0], [0.5, -0.5], [2.0, 1.0]], dtype=float)
    edge_features = np.zeros((3, 3, 2), dtype=float)
    edge_features[0, 1] = [0.2, 0.1]
    edge_features[0, 2] = [0.5, -0.2]
    edge_features[1, 0] = [0.2, 0.1]
    edge_features[2, 0] = [0.5, -0.2]

    out = st.edge_aware_message_passing(node_features, adjacency, edge_features, activation="relu")
    assert np.asarray(out).shape == node_features.shape

    stack = st.edge_feature_message_passing_stack(node_features, adjacency, edge_features, depth=3)
    assert np.asarray(stack.output).shape == node_features.shape
    assert len(stack.layers) == 3



def test_qkv_attention_and_transformer_stack() -> None:
    adjacency = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=float,
    )
    node_features = np.array(
        [[1.0, 0.1, -0.2], [0.5, -0.5, 0.3], [2.0, 1.0, -1.0], [-1.0, 0.2, 0.8]],
        dtype=float,
    )
    edge_features = np.ones((4, 4, 1), dtype=float) * 0.05

    qkv = st.qkv_graph_attention(node_features, adjacency, num_heads=3, edge_features=edge_features)
    assert len(qkv.heads) == 3
    assert np.asarray(qkv.output).shape == node_features.shape

    layer = st.graph_transformer_qkv_layer(node_features, adjacency, num_heads=2, edge_features=edge_features)
    assert np.asarray(layer.output).shape == node_features.shape
    assert layer.meta["explicit_qkv"] is True

    stack = st.graph_transformer_qkv_stack(node_features, adjacency, depth=2, num_heads=2, edge_features=edge_features)
    assert np.asarray(stack.output).shape == node_features.shape
    assert len(stack.layers) == 2



def test_wavelet_packet_2d_image() -> None:
    image = np.arange(64, dtype=float).reshape(8, 8)
    tree = st.wavelet_packet_2d_decompose(image, level=2, family="db2")
    recon = st.wavelet_packet_2d_reconstruct(tree)
    recon_arr = np.asarray(recon, dtype=float)
    assert recon_arr.ndim == 2
    assert recon_arr.shape[0] > 0 and recon_arr.shape[1] > 0
