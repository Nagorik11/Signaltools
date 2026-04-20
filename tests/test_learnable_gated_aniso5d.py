from __future__ import annotations

import numpy as np

import signaltools as st


def test_complex_learnable_tf_operator_and_stack() -> None:
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
    gain = np.ones((4, 2), dtype=np.complex128) * (0.8 + 0.1j)
    bias = np.zeros((4, 2), dtype=np.complex128)
    res = st.complex_learnable_tf_operator(features, frame_size=4, hop_size=2, gain=gain, bias=bias, activation="tanh")
    assert np.asarray(res.time_real).shape == features.shape
    assert np.asarray(res.stft_out_real).shape == (3, 4, 2)

    stacked = st.complex_learnable_tf_stack(features, depth=2, frame_size=4, hop_size=2, gain=gain, phase_step=0.1)
    assert stacked.meta["depth"] == 2
    assert np.asarray(stacked.magnitude_out).shape == (3, 4, 2)



def test_hybrid_graph_temporal_gated_memory_gru_and_lstm() -> None:
    sequence = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)
    adjacency = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=float,
    )
    edge_embeddings = np.ones((4, 4, 3), dtype=float) * 0.05

    gru = st.hybrid_graph_temporal_gated_memory(sequence, adjacency, edge_embeddings, num_heads=2, mode="gru")
    assert np.asarray(gru.output).shape == sequence.shape
    assert len(gru.gates["update"]) == sequence.shape[0]

    lstm = st.hybrid_graph_temporal_gated_stack(sequence, adjacency, edge_embeddings, depth=2, num_heads=2, mode="lstm")
    assert np.asarray(lstm.output).shape == sequence.shape
    assert lstm.meta["depth"] == 2
    assert lstm.cell_states is not None



def test_anisotropic_wavelet_packet_5d_with_rich_families() -> None:
    tensor5d = np.arange(2 * 2 * 4 * 4 * 4, dtype=float).reshape(2, 2, 4, 4, 4)
    tree = st.anisotropic_wavelet_packet_5d_decompose(
        tensor5d,
        level=1,
        families=("db8", "coif3", "bior97", "cdf97", "sym6"),
    )
    recon = st.anisotropic_wavelet_packet_5d_reconstruct(tree)
    recon_arr = np.asarray(recon, dtype=float)
    assert recon_arr.ndim == 5
    assert tree.meta["families"] == ["db8", "coif3", "bior97", "cdf97", "sym6"]
    assert tree.meta["family_kinds"] == ["orthogonal", "orthogonal", "biorthogonal", "biorthogonal", "orthogonal"]
    assert "db8" in tree.meta["supported"]
