from __future__ import annotations

import numpy as np

import signaltools as st


def test_edge_conditioned_convolution_and_stack() -> None:
    adjacency = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
    node_features = np.array([[1.0, 0.0, -0.5], [0.5, -0.5, 0.3], [2.0, 1.0, 0.1]], dtype=float)
    edge_features = np.zeros((3, 3, 2), dtype=float)
    edge_features[0, 1] = [0.2, 0.1]
    edge_features[0, 2] = [0.5, -0.2]
    edge_features[1, 0] = [0.2, 0.1]
    edge_features[2, 0] = [0.5, -0.2]

    out = st.edge_conditioned_convolution(node_features, adjacency, edge_features, activation="relu")
    assert np.asarray(out).shape == node_features.shape

    stack = st.edge_conditioned_conv_stack(node_features, adjacency, edge_features, depth=2)
    assert np.asarray(stack.output).shape == node_features.shape
    assert len(stack.layers) == 2



def test_masked_qkv_attention_and_transformer() -> None:
    adjacency = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=float,
    )
    node_features = np.array(
        [[1.0, 0.1, -0.2], [0.5, -0.5, 0.3], [2.0, 1.0, -1.0], [-1.0, 0.2, 0.8]],
        dtype=float,
    )
    edge_features = np.ones((4, 4, 2), dtype=float) * 0.05
    attention_mask = np.array(
        [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]],
        dtype=float,
    )

    qkv = st.masked_qkv_graph_attention(
        node_features,
        adjacency,
        num_heads=2,
        edge_features=edge_features,
        attention_mask=attention_mask,
        edge_bias_scale=1.5,
    )
    assert len(qkv.heads) == 2
    assert np.asarray(qkv.output).shape == node_features.shape
    assert qkv.meta["masked"] is True

    layer = st.graph_transformer_masked_qkv_layer(
        node_features,
        adjacency,
        num_heads=2,
        edge_features=edge_features,
        attention_mask=attention_mask,
        edge_bias_scale=1.5,
    )
    assert np.asarray(layer.output).shape == node_features.shape

    stack = st.graph_transformer_masked_qkv_stack(
        node_features,
        adjacency,
        depth=2,
        num_heads=2,
        edge_features=edge_features,
        attention_mask=attention_mask,
    )
    assert np.asarray(stack.output).shape == node_features.shape
    assert len(stack.layers) == 2



def test_wavelet_packet_3d_volume() -> None:
    volume = np.arange(4 * 4 * 4, dtype=float).reshape(4, 4, 4)
    tree = st.wavelet_packet_3d_decompose(volume, level=1, family="haar")
    recon = st.wavelet_packet_3d_reconstruct(tree)
    recon_arr = np.asarray(recon, dtype=float)
    assert recon_arr.ndim == 3
    assert recon_arr.shape[0] > 0 and recon_arr.shape[1] > 0 and recon_arr.shape[2] > 0
