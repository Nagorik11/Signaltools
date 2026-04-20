from __future__ import annotations

import numpy as np

import signaltools as st


def test_mode_conditioned_temporal_head_coupling_operator() -> None:
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
    causal = st.mode_conditioned_temporal_head_coupling_operator(features, frame_size=4, hop_size=2, num_heads=2, mode="causal")
    noncausal = st.mode_conditioned_temporal_head_coupling_operator(features, frame_size=4, hop_size=2, num_heads=2, mode="noncausal")
    assert np.asarray(causal.time_real).shape == features.shape
    assert np.asarray(noncausal.attention_weights).shape == (3, 2, 2)
    assert causal.meta["temporal_mode"] == "causal"
    assert noncausal.meta["temporal_mode"] == "noncausal"



def test_attentive_multiscale_hierarchical_hybrid_graph_temporal_gated_stack() -> None:
    sequence = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)
    adjacency = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=float,
    )
    edge_embeddings = np.ones((4, 4, 3), dtype=float) * 0.05
    out = st.attentive_multiscale_hierarchical_hybrid_graph_temporal_gated_stack(
        sequence, adjacency, edge_embeddings, depth=2, num_heads=2, scale_decays=(0.8, 0.95, 0.99)
    )
    assert np.asarray(out.output).shape == sequence.shape
    assert len(out.scale_attention_weights) == sequence.shape[0]
    assert out.meta["attentive_multiscale"] is True



def test_regularized_and_subband_attentive_wavelet_packet_5d() -> None:
    tensor5d = np.arange(2 * 2 * 4 * 4 * 4, dtype=float).reshape(2, 2, 4, 4, 4)
    sel = st.regularized_select_wavelet_family_per_axis_5d(
        tensor5d,
        candidate_families=["haar", "db8", "coif3", "cdf97"],
        block_shape=(1, 1, 2, 2, 2),
        complexity_lambda=0.1,
    )
    assert len(sel["families"]) == 5

    reg_tree = st.regularized_adaptive_wavelet_packet_5d_decompose(
        tensor5d,
        level=1,
        candidate_families=["haar", "db8", "coif3", "cdf97"],
        block_shape=(1, 1, 2, 2, 2),
        complexity_lambda=0.1,
    )
    reg_recon = st.regularized_adaptive_wavelet_packet_5d_reconstruct(reg_tree)
    assert np.asarray(reg_recon, dtype=float).ndim == 5
    assert reg_tree.meta["adaptive_regularized"] is True

    att_tree = st.subband_attentive_wavelet_packet_5d_decompose(
        tensor5d,
        level=1,
        candidate_families=["haar", "db8", "coif3", "cdf97"],
        block_shape=(1, 1, 2, 2, 2),
    )
    att_recon = st.subband_attentive_wavelet_packet_5d_reconstruct(att_tree)
    assert np.asarray(att_recon, dtype=float).ndim == 5
    assert att_tree.meta["attentive_subbands"] is True
