from __future__ import annotations

import numpy as np

import signaltools as st


def test_complex_spectral_operators() -> None:
    features = np.array([
        [1.0 + 0.0j, 0.0 + 1.0j],
        [0.5 + 0.5j, -0.5 + 0.25j],
        [0.0 - 1.0j, 1.0 + 0.0j],
        [-0.25 + 0.75j, 0.1 - 0.2j],
    ], dtype=np.complex128)
    dft = st.complex_dft_multichannel(features)
    assert np.asarray(dft.real).shape == features.shape

    mask = np.ones(features.shape, dtype=np.complex128)
    masked = st.complex_spectral_mask(features, mask=mask, phase_shift=0.2)
    shifted = st.complex_spectral_shift(features, bins=1)
    assert np.asarray(masked.imag).shape == features.shape
    assert np.asarray(shifted.magnitude).shape == features.shape



def test_hybrid_node_edge_temporal_attention() -> None:
    sequence = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)
    adjacency = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=float,
    )
    edge_embeddings = np.ones((4, 4, 3), dtype=float) * 0.05

    hybrid = st.hybrid_node_edge_temporal_attention(sequence, adjacency, edge_embeddings, num_heads=2, temporal_window=1)
    assert np.asarray(hybrid.output).shape == sequence.shape
    assert len(hybrid.temporal_weights) == sequence.shape[0]

    stack = st.hybrid_graph_temporal_transformer_stack(sequence, adjacency, edge_embeddings, depth=2, num_heads=2)
    assert np.asarray(stack.output).shape == sequence.shape
    assert stack.meta["depth"] == 2



def test_anisotropic_wavelet_packet_4d() -> None:
    tensor4d = np.arange(2 * 4 * 4 * 4, dtype=float).reshape(2, 4, 4, 4)
    tree = st.anisotropic_wavelet_packet_4d_decompose(
        tensor4d,
        level=1,
        families=("haar", "db2", "cdf53", "coif1"),
    )
    recon = st.anisotropic_wavelet_packet_4d_reconstruct(tree)
    recon_arr = np.asarray(recon, dtype=float)
    assert recon_arr.ndim == 4
    assert tree.meta["families"] == ["haar", "db2", "cdf53", "coif1"]
