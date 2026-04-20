from __future__ import annotations

import numpy as np

import signaltools as st


def test_channel_mix_and_positional_encodings() -> None:
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    features = np.array([[1.0, 0.0], [0.5, -0.5], [2.0, 1.0]], dtype=float)

    mixed = st.channel_mix(features, out_channels=3, activation="relu")
    assert np.asarray(mixed.output).shape == (3, 3)

    lap = np.asarray(st.laplacian_positional_encoding(adjacency, dimensions=2), dtype=float)
    rw = np.asarray(st.random_walk_positional_encoding(adjacency, steps=3), dtype=float)
    aug = np.asarray(st.augment_with_graph_positional_encoding(features, adjacency, method="laplacian", dimensions=2), dtype=float)

    assert lap.shape[0] == adjacency.shape[0]
    assert rw.shape == (3, 3)
    assert aug.shape == (3, 4)



def test_enhanced_transformer_with_positional_and_channel_mix() -> None:
    adjacency = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=float,
    )
    features = np.array(
        [[1.0, 0.1, -0.2], [0.5, -0.5, 0.3], [2.0, 1.0, -1.0], [-1.0, 0.2, 0.8]],
        dtype=float,
    )
    edge_features = np.ones((4, 4, 1), dtype=float) * 0.05
    attention_mask = np.array(
        [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]],
        dtype=float,
    )

    layer = st.graph_transformer_enhanced_layer(
        features,
        adjacency,
        num_heads=2,
        edge_features=edge_features,
        attention_mask=attention_mask,
        positional_encoding_method="laplacian",
        positional_dimensions=2,
        mix_strength=0.2,
    )
    assert np.asarray(layer.output).shape == features.shape
    assert layer.meta["enhanced"] is True

    stack = st.graph_transformer_enhanced_stack(
        features,
        adjacency,
        depth=2,
        num_heads=2,
        edge_features=edge_features,
        attention_mask=attention_mask,
        positional_encoding_method="random_walk",
        positional_steps=2,
        mix_strength=0.2,
    )
    assert np.asarray(stack.output).shape == features.shape
    assert len(stack.layers) == 2



def test_anisotropic_wavelet_packet_3d() -> None:
    volume = np.arange(4 * 4 * 4, dtype=float).reshape(4, 4, 4)
    tree = st.anisotropic_wavelet_packet_3d_decompose(volume, level=1, families=("haar", "db2", "cdf53"))
    recon = st.anisotropic_wavelet_packet_3d_reconstruct(tree)
    recon_arr = np.asarray(recon, dtype=float)
    assert recon_arr.ndim == 3
    assert tree.meta["families"] == ["haar", "db2", "cdf53"]
