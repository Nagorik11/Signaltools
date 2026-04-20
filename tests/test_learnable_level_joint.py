from __future__ import annotations

import numpy as np

import signaltools as st


def test_joint_temporal_spectral_regularized_coupling_operator() -> None:
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
    out = st.joint_temporal_spectral_regularized_coupling_operator(features, frame_size=4, hop_size=2, num_heads=2, temporal_lambda=0.4, spectral_lambda=0.2)
    assert np.asarray(out.time_real).shape == features.shape
    assert out.meta["joint_temporal_spectral_regularized"] is True


def test_learnable_multiobjective_and_level_attentive_trees() -> None:
    tensor5d = np.arange(2 * 2 * 4 * 4 * 4, dtype=float).reshape(2, 2, 4, 4, 4)
    tree = st.learnable_multiobjective_wavelet_packet_5d_decompose(
        tensor5d,
        level=1,
        candidate_families=["haar", "db8", "coif3", "cdf97"],
        block_shape=(1, 1, 2, 2, 2),
        steps=2,
        step_size=0.2,
    )
    recon = st.learnable_multiobjective_wavelet_packet_5d_reconstruct(tree)
    assert np.asarray(recon, dtype=float).ndim == 5
    assert tree.meta["learnable_multiobjective"] is True

    level_tree = st.level_attentive_wavelet_packet_5d_decompose(
        tensor5d,
        level=1,
        candidate_families=["haar", "db8", "coif3", "cdf97"],
        block_shape=(1, 1, 2, 2, 2),
    )
    level_recon = st.level_attentive_wavelet_packet_5d_reconstruct(level_tree)
    assert np.asarray(level_recon, dtype=float).ndim == 5
    assert level_tree.meta["level_attentive"] is True
