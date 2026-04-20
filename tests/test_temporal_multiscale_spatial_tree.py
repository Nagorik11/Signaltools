from __future__ import annotations

import numpy as np

import signaltools as st


def test_temporal_complex_head_coupling_operator() -> None:
    features = np.array([
        [1.0 + 0.0j, 0.0 + 1.0j],
        [0.5 + 0.5j, -0.5 + 0.25j],
        [0.0 - 1.0j, 1.0 + 0.0j],
        [-0.25 + 0.75j, 0.1 - 0.2j],
        [0.2 + 0.1j, -0.1 + 0.3j],
        [0.1 - 0.2j, 0.4 + 0.2j],
        [0.0 + 0.0j, 0.5 - 0.5j],
        [0.3 + 0.6j, -0.2 + 0.1j],
    ], dtype=np.complex128)
    out = st.temporal_complex_head_coupling_operator(features, frame_size=4, hop_size=2, num_heads=2)
    assert np.asarray(out.time_real).shape == features.shape
    assert np.asarray(out.attention_weights).shape == (3, 2, 2)
    assert out.meta["time_dependent"] is True



def test_multiscale_hierarchical_hybrid_graph_temporal_gated_stack() -> None:
    sequence = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)
    adjacency = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=float,
    )
    edge_embeddings = np.ones((4, 4, 3), dtype=float) * 0.05
    out = st.multiscale_hierarchical_hybrid_graph_temporal_gated_memory(
        sequence, adjacency, edge_embeddings, num_heads=2, scale_decays=(0.8, 0.95, 0.99)
    )
    assert np.asarray(out.output).shape == sequence.shape
    assert len(out.scale_states["scale_0"]) == sequence.shape[0]

    stacked = st.multiscale_hierarchical_hybrid_graph_temporal_gated_stack(
        sequence, adjacency, edge_embeddings, depth=2, num_heads=2, scale_decays=(0.8, 0.95, 0.99)
    )
    assert np.asarray(stacked.output).shape == sequence.shape
    assert stacked.meta["multiscale"] is True
    assert stacked.meta["depth"] == 2



def test_spatially_variable_wavelet_packet_5d_tree() -> None:
    tensor5d = np.arange(2 * 2 * 4 * 4 * 4, dtype=float).reshape(2, 2, 4, 4, 4)
    tree = st.spatially_variable_wavelet_packet_5d_decompose(
        tensor5d,
        level=1,
        candidate_families=["haar", "db8", "coif3", "cdf97"],
        block_shape=(1, 1, 2, 2, 2),
    )
    recon = st.spatially_variable_wavelet_packet_5d_reconstruct(tree)
    recon_arr = np.asarray(recon, dtype=float)
    assert recon_arr.ndim == 5
    assert tree.meta["adaptive_spatial"] is True
    assert "" in tree.meta["node_families"]
