from __future__ import annotations

import numpy as np

import signaltools as st


def test_multihead_band_complex_tf_operator_and_stack() -> None:
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
    head_gains = np.array([0.8 + 0.1j, 1.1 - 0.05j], dtype=np.complex128)
    res = st.multihead_band_complex_tf_operator(features, frame_size=4, hop_size=2, num_heads=2, head_gains=head_gains)
    assert np.asarray(res.time_real).shape == features.shape
    assert np.asarray(res.head_outputs_real).shape[:2] == (2, 3)
    assert len(res.band_assignments) == 4

    stacked = st.multihead_band_complex_tf_stack(features, depth=2, frame_size=4, hop_size=2, num_heads=2, head_gains=head_gains)
    assert stacked.meta["depth"] == 2



def test_bidirectional_hybrid_graph_temporal_gated_stack() -> None:
    sequence = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)
    adjacency = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=float,
    )
    edge_embeddings = np.ones((4, 4, 3), dtype=float) * 0.05
    out = st.bidirectional_hybrid_graph_temporal_gated_memory(sequence, adjacency, edge_embeddings, num_heads=2, mode="gru")
    assert np.asarray(out.output).shape == sequence.shape
    assert out.meta["bidirectional"] is True

    stacked = st.bidirectional_hybrid_graph_temporal_gated_stack(sequence, adjacency, edge_embeddings, depth=2, num_heads=2, mode="lstm", merge="mean")
    assert np.asarray(stacked.output).shape == sequence.shape
    assert stacked.meta["depth"] == 2



def test_adaptive_wavelet_packet_5d_selection_and_reconstruction() -> None:
    tensor5d = np.arange(2 * 2 * 4 * 4 * 4, dtype=float).reshape(2, 2, 4, 4, 4)
    selected = st.select_wavelet_family_per_axis_5d(tensor5d, candidate_families=["haar", "db8", "coif3", "cdf97"])
    assert len(selected["families"]) == 5
    assert selected["criterion"] == "low_high_ratio"

    tree = st.adaptive_wavelet_packet_5d_decompose(tensor5d, level=1, candidate_families=["haar", "db8", "coif3", "cdf97"])
    recon = st.adaptive_wavelet_packet_5d_reconstruct(tree)
    recon_arr = np.asarray(recon, dtype=float)
    assert recon_arr.ndim == 5
    assert tree.meta["adaptive"] is True
    assert len(tree.meta["families"]) == 5
