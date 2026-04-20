from __future__ import annotations

import numpy as np
import pytest

import signaltools as st


def test_more_wavelet_families_and_kinds() -> None:
    families = st.available_wavelet_families()
    for family in ["db8", "sym6", "coif3", "bior53", "bior97"]:
        assert family in families
    assert st.wavelet_family_kind("db8") == "orthogonal"
    assert st.wavelet_family_kind("bior53") == "biorthogonal"



def test_biorthogonal_wavelet_packet_runs() -> None:
    x = np.cos(2 * np.pi * 0.075 * np.arange(64)).tolist()
    for family in ["bior53", "bior97", "coif3"]:
        tree = st.wavelet_packet_decompose(x, level=2, family=family)
        recon = st.wavelet_packet_reconstruct(tree)
        assert len(recon) > 0
        assert tree.meta["kind"] in {"orthogonal", "biorthogonal"}



def test_graph_block_normalize_modes() -> None:
    x = [1.0, 2.0, 3.0, 4.0]
    for mode in ["layernorm", "batchnorm", "zscore", "l2", "none"]:
        y = st.graph_block_normalize(x, mode=mode, gamma=1.2, beta=0.1)
        assert len(y) == len(x)
    with pytest.raises(ValueError):
        st.graph_block_normalize(x, mode="weird")



def test_multihead_graph_attention_outputs() -> None:
    adjacency = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
    signal = [1.0, -0.5, 2.0]
    result = st.multihead_graph_attention(signal, adjacency, num_heads=3, alpha=0.9)
    assert len(result.heads) == 3
    assert len(result.output) == len(signal)
    for mat in result.attention_matrices:
        m = np.asarray(mat, dtype=float)
        assert np.allclose(m.sum(axis=1), 1.0)

    concat_result = st.multihead_graph_attention(signal, adjacency, num_heads=2, concat=True)
    assert len(concat_result.output) == 2 * len(signal)

    with pytest.raises(ValueError):
        st.multihead_graph_attention(signal, adjacency, num_heads=2, value_scales=[1.0])



def test_deep_gnn_stack_with_multihead_attention_and_batchnorm() -> None:
    adjacency = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=float,
    )
    signal = [1.0, -1.0, 2.0, -2.0]
    result = st.deep_gnn_stack(
        signal,
        adjacency,
        layer_weights=[[1.0, -0.2], [0.8, 0.15], [1.0, 0.05, -0.02]],
        activation="gelu",
        normalization="batchnorm",
        attention=True,
        num_heads=4,
        attention_alpha=0.75,
        attention_mix=0.4,
        norm_gamma=1.1,
        norm_beta=0.05,
        pooling_factor=2,
        residual=True,
    )
    assert len(result.layers) == 3
    assert len(result.attention_matrices) == 3
    assert len(result.pooled) == 3
    assert result.meta["num_heads"] == 4
    assert result.meta["normalization"] == "batchnorm"

    with pytest.raises(ValueError):
        st.deep_gnn_stack(signal, adjacency, [[1.0]], attention=True, num_heads=2, concat_heads=True)
