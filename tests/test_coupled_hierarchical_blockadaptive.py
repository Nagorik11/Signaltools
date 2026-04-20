from __future__ import annotations

import numpy as np

import signaltools as st


def test_complex_multiband_head_coupling_operator_and_stack() -> None:
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
    res = st.complex_multiband_head_coupling_operator(
        features,
        frame_size=4,
        hop_size=2,
        num_heads=2,
        head_gains=np.array([0.9 + 0.1j, 1.1 - 0.05j], dtype=np.complex128),
        coupling_matrix=np.array([[1.0, 0.2], [0.1, 1.0]], dtype=float),
    )
    assert np.asarray(res.time_real).shape == features.shape
    assert np.asarray(res.coupled_specs_real).shape[:3] == (2, 3, 4)
    assert np.asarray(res.attention_weights).shape == (2, 2)

    stacked = st.complex_multiband_head_coupling_stack(
        features,
        depth=2,
        frame_size=4,
        hop_size=2,
        num_heads=2,
        head_gains=np.array([0.9 + 0.1j, 1.1 - 0.05j], dtype=np.complex128),
    )
    assert stacked.meta["depth"] == 2



def test_hierarchical_hybrid_graph_temporal_gated_stack() -> None:
    sequence = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)
    adjacency = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=float,
    )
    edge_embeddings = np.ones((4, 4, 3), dtype=float) * 0.05

    out = st.hierarchical_hybrid_graph_temporal_gated_memory(sequence, adjacency, edge_embeddings, num_heads=2, mode="gru")
    assert np.asarray(out.output).shape == sequence.shape
    assert len(out.fast_states) == sequence.shape[0]
    assert len(out.slow_states) == sequence.shape[0]

    stacked = st.hierarchical_hybrid_graph_temporal_gated_stack(sequence, adjacency, edge_embeddings, depth=2, num_heads=2, mode="lstm")
    assert np.asarray(stacked.output).shape == sequence.shape
    assert stacked.meta["depth"] == 2
    assert stacked.meta["hierarchical"] is True



def test_blockwise_adaptive_wavelet_packet_5d() -> None:
    tensor5d = np.arange(2 * 2 * 4 * 4 * 4, dtype=float).reshape(2, 2, 4, 4, 4)
    report = st.select_wavelet_family_per_axis_5d_local_blocks(
        tensor5d,
        candidate_families=["haar", "db8", "coif3", "cdf97"],
        block_shape=(1, 1, 2, 2, 2),
    )
    assert len(report["families"]) == 5
    assert report["criterion"] == "local_variation"
    assert len(report["blocks"]) > 0

    tree = st.adaptive_blockwise_wavelet_packet_5d_decompose(
        tensor5d,
        level=1,
        candidate_families=["haar", "db8", "coif3", "cdf97"],
        block_shape=(1, 1, 2, 2, 2),
    )
    recon = st.adaptive_blockwise_wavelet_packet_5d_reconstruct(tree)
    recon_arr = np.asarray(recon, dtype=float)
    assert recon_arr.ndim == 5
    assert tree.meta["adaptive_blockwise"] is True
