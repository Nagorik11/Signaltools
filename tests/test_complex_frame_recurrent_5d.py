from __future__ import annotations

import numpy as np

import signaltools as st


def test_complex_stft_and_frame_operator() -> None:
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
    res = st.complex_stft_multichannel(features, frame_size=4, hop_size=2)
    assert np.asarray(res.stft_real).shape == (3, 4, 2)
    mask = np.ones((4, 2), dtype=np.complex128)
    out = st.complex_frame_operator(features, frame_size=4, hop_size=2, phase_shift=0.2, spectral_mask=mask)
    assert np.asarray(out.time_real).shape == features.shape
    assert out.meta["frames"] == 3



def test_recurrent_hybrid_attention_stack() -> None:
    sequence = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)
    adjacency = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=float,
    )
    edge_embeddings = np.ones((4, 4, 3), dtype=float) * 0.05

    hybrid = st.recurrent_hybrid_node_edge_temporal_attention(sequence, adjacency, edge_embeddings, num_heads=2, temporal_window=1)
    assert np.asarray(hybrid.output).shape == sequence.shape
    assert len(hybrid.memory_states) == sequence.shape[0]

    stack = st.recurrent_hybrid_graph_temporal_transformer_stack(sequence, adjacency, edge_embeddings, depth=2, num_heads=2)
    assert np.asarray(stack.output).shape == sequence.shape
    assert stack.meta["depth"] == 2
    assert stack.meta["recurrent"] is True



def test_wavelet_packet_5d_roundtrip_shape() -> None:
    tensor5d = np.arange(2 * 2 * 4 * 4 * 4, dtype=float).reshape(2, 2, 4, 4, 4)
    tree = st.wavelet_packet_5d_decompose(tensor5d, level=1, family="haar")
    recon = st.wavelet_packet_5d_reconstruct(tree)
    recon_arr = np.asarray(recon, dtype=float)
    assert recon_arr.ndim == 5
    assert tree.meta["wavelet"] == "haar"

    aniso = st.anisotropic_wavelet_packet_5d_decompose(
        tensor5d,
        level=1,
        families=("haar", "db2", "cdf53", "coif1", "sym2"),
    )
    recon2 = st.anisotropic_wavelet_packet_5d_reconstruct(aniso)
    recon2_arr = np.asarray(recon2, dtype=float)
    assert recon2_arr.ndim == 5
    assert aniso.meta["families"] == ["haar", "db2", "cdf53", "coif1", "sym2"]
