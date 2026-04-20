from __future__ import annotations

import numpy as np

import signaltools as st


def test_stability_regularized_temporal_head_coupling_operator() -> None:
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
    out = st.stability_regularized_temporal_head_coupling_operator(features, frame_size=4, hop_size=2, num_heads=2, stability_lambda=0.4)
    assert np.asarray(out.time_real).shape == features.shape
    assert out.meta["temporal_stability_regularized"] is True


def test_cross_branch_attentive_wavelet_packet_5d() -> None:
    tensor5d = np.arange(2 * 2 * 4 * 4 * 4, dtype=float).reshape(2, 2, 4, 4, 4)
    tree = st.cross_branch_attentive_wavelet_packet_5d_decompose(
        tensor5d,
        level=1,
        candidate_families=["haar", "db8", "coif3", "cdf97"],
        block_shape=(1, 1, 2, 2, 2),
    )
    recon = st.cross_branch_attentive_wavelet_packet_5d_reconstruct(tree)
    assert np.asarray(recon, dtype=float).ndim == 5
    assert tree.meta["cross_branch_attention"] is True


def test_weighted_multiobjective_wavelet_packet_5d() -> None:
    tensor5d = np.arange(2 * 2 * 4 * 4 * 4, dtype=float).reshape(2, 2, 4, 4, 4)
    sel = st.weighted_multiobjective_select_wavelet_family_per_axis_5d(
        tensor5d,
        candidate_families=["haar", "db8", "coif3", "cdf97"],
        block_shape=(1, 1, 2, 2, 2),
        axis_weights=[1.0, 0.5, 1.5, 1.0, 0.75],
        precision_weight=1.0,
        cost_weight=0.05,
        stability_weight=0.05,
    )
    assert len(sel["families"]) == 5

    tree = st.weighted_multiobjective_wavelet_packet_5d_decompose(
        tensor5d,
        level=1,
        candidate_families=["haar", "db8", "coif3", "cdf97"],
        block_shape=(1, 1, 2, 2, 2),
        axis_weights=[1.0, 0.5, 1.5, 1.0, 0.75],
        precision_weight=1.0,
        cost_weight=0.05,
        stability_weight=0.05,
    )
    recon = st.weighted_multiobjective_wavelet_packet_5d_reconstruct(tree)
    assert np.asarray(recon, dtype=float).ndim == 5
    assert tree.meta["adaptive_multiobjective"] is True
