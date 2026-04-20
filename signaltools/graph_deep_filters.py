"""Graph scattering and deeper spectral GNN-style layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .graph_filters import graph_laplacian
from .graph_wavelets import graph_wavelet_transform
from .graph_positional import augment_with_graph_positional_encoding
from .utils import ensure_positive_int, to_1d_float_array


@dataclass
class GNNStackResult:
    output: list[float]
    layers: list[list[float]]
    pooled: list[list[float]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MultiHeadAttentionResult:
    output: list[float]
    heads: list[list[float]]
    attention_matrices: list[list[list[float]]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MultiHeadNodeAttentionResult:
    output: list[list[float]]
    heads: list[list[list[float]]]
    attention_matrices: list[list[list[float]]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeepGNNResult:
    output: list[float]
    layers: list[list[float]]
    pooled: list[list[float]]
    attention_matrices: list[list[list[float]]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphTransformerResult:
    output: list[list[float]]
    layers: list[list[list[float]]]
    attention_matrices: list[list[list[float]]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _apply_activation(x: np.ndarray, activation: str) -> np.ndarray:
    if activation == "relu":
        return np.maximum(x, 0.0)
    if activation == "tanh":
        return np.tanh(x)
    if activation == "gelu":
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    if activation == "linear":
        return x
    raise ValueError("Unsupported activation")



def _as_node_features(features: list[list[float]] | list[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("features must be a 2D array-like of shape (nodes, channels)")
    return x



def graph_block_normalize(
    signal: list[float] | list[int] | np.ndarray,
    mode: str = "layernorm",
    gamma: float = 1.0,
    beta: float = 0.0,
    eps: float = 1e-5,
) -> list[float]:
    """Normalize a 1D graph feature block with optional affine parameters."""
    x = to_1d_float_array(signal)
    if mode == "none":
        return x.tolist()
    if mode == "l2":
        norm = float(np.linalg.norm(x))
        y = x if norm <= eps else x / norm
        return (gamma * y + beta).tolist()
    if mode in {"zscore", "layernorm", "batchnorm"}:
        mean = float(np.mean(x))
        std = float(np.std(x))
        y = (x - mean) / max(std, eps)
        return (gamma * y + beta).tolist()
    raise ValueError("Unsupported normalization mode")



def graph_block_normalize_multichannel(
    features: list[list[float]] | list[float] | np.ndarray,
    mode: str = "layernorm",
    gamma: float = 1.0,
    beta: float = 0.0,
    eps: float = 1e-5,
) -> list[list[float]]:
    """Normalize node features of shape (nodes, channels)."""
    x = _as_node_features(features)
    if mode == "none":
        return x.tolist()
    if mode == "l2":
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        y = x / np.maximum(norms, eps)
        return (gamma * y + beta).tolist()
    if mode == "layernorm":
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True)
        y = (x - mean) / np.maximum(std, eps)
        return (gamma * y + beta).tolist()
    if mode in {"batchnorm", "zscore"}:
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)
        y = (x - mean) / np.maximum(std, eps)
        return (gamma * y + beta).tolist()
    raise ValueError("Unsupported normalization mode")



def graph_pool(signal: list[float] | list[int], factor: int = 2, mode: str = "mean") -> list[float]:
    x = to_1d_float_array(signal)
    factor = ensure_positive_int(factor, "factor")
    if x.size == 0:
        return []
    pooled: list[float] = []
    for i in range(0, len(x), factor):
        block = x[i : i + factor]
        if mode == "mean":
            pooled.append(float(np.mean(block)))
        elif mode == "max":
            pooled.append(float(np.max(block)))
        else:
            raise ValueError("Unsupported pooling mode")
    return pooled



def graph_attention_matrix(
    signal: list[float] | list[int],
    adjacency: list[list[float]] | np.ndarray,
    alpha: float = 1.0,
    add_self_loops: bool = True,
) -> list[list[float]]:
    """Build a simple row-normalized attention matrix over graph neighbors."""
    x = to_1d_float_array(signal)
    A = np.asarray(adjacency, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if A.shape[0] != len(x):
        raise ValueError("signal length must match graph size")
    A = A.copy()
    if add_self_loops:
        A = A + np.eye(A.shape[0], dtype=np.float64)
    scores = np.zeros_like(A, dtype=np.float64)
    for i in range(A.shape[0]):
        mask = A[i] > 0
        if np.any(mask):
            row_scores = alpha * x[i] * x
            valid_scores = row_scores[mask]
            valid_scores = valid_scores - np.max(valid_scores)
            exp_scores = np.exp(valid_scores)
            scores[i, mask] = exp_scores / max(np.sum(exp_scores), 1e-12)
        else:
            scores[i, i] = 1.0
    return scores.tolist()



def graph_attention_filter(
    signal: list[float] | list[int],
    adjacency: list[list[float]] | np.ndarray,
    alpha: float = 1.0,
    add_self_loops: bool = True,
    activation: str = "linear",
) -> list[float]:
    x = to_1d_float_array(signal)
    attn = np.asarray(graph_attention_matrix(x.tolist(), adjacency, alpha=alpha, add_self_loops=add_self_loops), dtype=np.float64)
    y = attn @ x
    return _apply_activation(y, activation).tolist()



def multihead_graph_attention(
    signal: list[float] | list[int],
    adjacency: list[list[float]] | np.ndarray,
    num_heads: int = 4,
    alpha: float = 1.0,
    value_scales: list[float] | None = None,
    concat: bool = False,
    add_self_loops: bool = True,
    activation: str = "linear",
) -> MultiHeadAttentionResult:
    """Apply multi-head scalar graph attention to a 1D graph signal."""
    x = to_1d_float_array(signal)
    num_heads = ensure_positive_int(num_heads, "num_heads")
    if value_scales is None:
        value_scales = [1.0 + 0.15 * head for head in range(num_heads)]
    if len(value_scales) != num_heads:
        raise ValueError("value_scales length must match num_heads")

    head_outputs: list[list[float]] = []
    head_matrices: list[list[list[float]]] = []
    for head in range(num_heads):
        head_alpha = alpha * (1.0 + 0.1 * head)
        attn = np.asarray(
            graph_attention_matrix(x.tolist(), adjacency, alpha=head_alpha, add_self_loops=add_self_loops),
            dtype=np.float64,
        )
        values = float(value_scales[head]) * x
        y = attn @ values
        y = _apply_activation(y, activation)
        head_outputs.append(y.tolist())
        head_matrices.append(attn.tolist())

    stacked = np.asarray(head_outputs, dtype=np.float64)
    output = stacked.reshape(-1) if concat else np.mean(stacked, axis=0)
    return MultiHeadAttentionResult(
        output=output.tolist(),
        heads=head_outputs,
        attention_matrices=head_matrices,
        meta={
            "num_heads": num_heads,
            "alpha": alpha,
            "concat": concat,
            "add_self_loops": add_self_loops,
            "activation": activation,
        },
    )



def multihead_graph_attention_multichannel(
    features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    num_heads: int = 4,
    alpha: float = 1.0,
    concat: bool = False,
    add_self_loops: bool = True,
    activation: str = "linear",
) -> MultiHeadNodeAttentionResult:
    """Apply multi-head attention to node features of shape (nodes, channels)."""
    x = _as_node_features(features)
    num_heads = ensure_positive_int(num_heads, "num_heads")
    A = np.asarray(adjacency, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if A.shape[0] != x.shape[0]:
        raise ValueError("node count must match graph size")

    head_outputs: list[list[list[float]]] = []
    head_matrices: list[list[list[float]]] = []
    for head in range(num_heads):
        scale_q = 1.0 + 0.1 * head
        scale_v = 1.0 + 0.15 * head
        q = np.mean(x * scale_q, axis=1)
        attn = np.asarray(graph_attention_matrix(q.tolist(), A, alpha=alpha * scale_q, add_self_loops=add_self_loops), dtype=np.float64)
        values = x * scale_v
        out = attn @ values
        out = _apply_activation(out, activation)
        head_outputs.append(out.tolist())
        head_matrices.append(attn.tolist())

    stacked = np.asarray(head_outputs, dtype=np.float64)  # (heads, nodes, channels)
    output = np.concatenate([stacked[h] for h in range(num_heads)], axis=1) if concat else np.mean(stacked, axis=0)
    return MultiHeadNodeAttentionResult(
        output=output.tolist(),
        heads=head_outputs,
        attention_matrices=head_matrices,
        meta={
            "num_heads": num_heads,
            "alpha": alpha,
            "concat": concat,
            "add_self_loops": add_self_loops,
            "activation": activation,
            "channels_in": int(x.shape[1]),
            "channels_out": int(output.shape[1]),
        },
    )



def spectral_gnn_filter(
    signal: list[float] | list[int],
    adjacency: list[list[float]] | np.ndarray,
    weights: list[float],
    normalized: bool = True,
    activation: str = "relu",
) -> list[float]:
    x = to_1d_float_array(signal)
    L = graph_laplacian(adjacency, normalized=normalized)
    if len(x) != L.shape[0]:
        raise ValueError("signal length must match graph size")
    y = np.zeros_like(x)
    current = x.copy()
    for k, w in enumerate(weights):
        if k == 0:
            current = x
        elif k == 1:
            current = L @ x
        else:
            current = L @ current
        y += float(w) * current
    return _apply_activation(y, activation).tolist()



def spectral_gnn_filter_multichannel(
    features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    weights: list[float],
    normalized: bool = True,
    activation: str = "relu",
) -> list[list[float]]:
    """Apply the same graph polynomial filter independently to each node channel."""
    x = _as_node_features(features)
    L = graph_laplacian(adjacency, normalized=normalized)
    if x.shape[0] != L.shape[0]:
        raise ValueError("node count must match graph size")
    y = np.zeros_like(x)
    current = x.copy()
    for k, w in enumerate(weights):
        if k == 0:
            current = x
        elif k == 1:
            current = L @ x
        else:
            current = L @ current
        y += float(w) * current
    return _apply_activation(y, activation).tolist()



def graph_scattering_transform(signal: list[float] | list[int], adjacency: list[list[float]] | np.ndarray, scales: list[float], normalized: bool = True) -> list[list[float]]:
    x = to_1d_float_array(signal)
    wavelets = graph_wavelet_transform(x.tolist(), adjacency, scales=scales, normalized=normalized, kind="heat")
    return [np.abs(np.asarray(coeffs, dtype=np.float64)).tolist() for coeffs in wavelets]



def stacked_gnn(
    signal: list[float] | list[int],
    adjacency: list[list[float]] | np.ndarray,
    layer_weights: list[list[float]],
    activation: str = "relu",
    pooling_factor: int = 1,
    pooling_mode: str = "mean",
    residual: bool = False,
) -> GNNStackResult:
    x = to_1d_float_array(signal)
    layers: list[list[float]] = []
    pooled: list[list[float]] = []
    current = x.tolist()
    for weights in layer_weights:
        out = spectral_gnn_filter(current, adjacency, weights, activation=activation)
        if residual and len(out) == len(current):
            out = (np.asarray(out, dtype=np.float64) + np.asarray(current, dtype=np.float64)).tolist()
        layers.append(out)
        if pooling_factor > 1:
            pooled.append(graph_pool(out, factor=pooling_factor, mode=pooling_mode))
        current = out
    return GNNStackResult(output=current, layers=layers, pooled=pooled, meta={"activation": activation, "pooling_factor": pooling_factor, "pooling_mode": pooling_mode, "residual": residual})



def deep_gnn_stack(
    signal: list[float] | list[int],
    adjacency: list[list[float]] | np.ndarray,
    layer_weights: list[list[float]],
    activation: str = "relu",
    normalization: str = "layernorm",
    attention: bool = False,
    attention_alpha: float = 1.0,
    attention_mix: float = 0.5,
    num_heads: int = 1,
    concat_heads: bool = False,
    norm_gamma: float = 1.0,
    norm_beta: float = 0.0,
    norm_eps: float = 1e-5,
    pooling_factor: int = 1,
    pooling_mode: str = "mean",
    residual: bool = False,
) -> DeepGNNResult:
    """Apply stacked spectral graph layers with optional multi-head attention and normalization."""
    x = to_1d_float_array(signal)
    current = x.copy()
    layers: list[list[float]] = []
    pooled: list[list[float]] = []
    attention_matrices: list[list[list[float]]] = []

    if not 0.0 <= attention_mix <= 1.0:
        raise ValueError("attention_mix must be between 0 and 1")
    if concat_heads and num_heads > 1:
        raise ValueError("concat_heads is not supported inside deep_gnn_stack because it changes feature length")

    for weights in layer_weights:
        base = np.asarray(spectral_gnn_filter(current.tolist(), adjacency, weights, activation="linear"), dtype=np.float64)
        attn_matrix = np.eye(len(base), dtype=np.float64)
        if attention:
            if num_heads > 1:
                mh = multihead_graph_attention(current.tolist(), adjacency, num_heads=num_heads, alpha=attention_alpha, concat=False, activation="linear")
                attn_out = np.asarray(mh.output, dtype=np.float64)
                attn_matrix = np.mean(np.asarray(mh.attention_matrices, dtype=np.float64), axis=0)
            else:
                attn_matrix = np.asarray(graph_attention_matrix(current.tolist(), adjacency, alpha=attention_alpha), dtype=np.float64)
                attn_out = attn_matrix @ current
            out = (1.0 - attention_mix) * base + attention_mix * attn_out
        else:
            out = base
        out = np.asarray(graph_block_normalize(out, mode=normalization, gamma=norm_gamma, beta=norm_beta, eps=norm_eps), dtype=np.float64)
        out = _apply_activation(out, activation)
        if residual and out.shape == current.shape:
            out = out + current
        layers.append(out.tolist())
        attention_matrices.append(attn_matrix.tolist())
        if pooling_factor > 1:
            pooled.append(graph_pool(out.tolist(), factor=pooling_factor, mode=pooling_mode))
        current = out

    return DeepGNNResult(
        output=current.tolist(),
        layers=layers,
        pooled=pooled,
        attention_matrices=attention_matrices,
        meta={
            "activation": activation,
            "normalization": normalization,
            "norm_gamma": norm_gamma,
            "norm_beta": norm_beta,
            "norm_eps": norm_eps,
            "attention": attention,
            "attention_alpha": attention_alpha,
            "attention_mix": attention_mix,
            "num_heads": num_heads,
            "pooling_factor": pooling_factor,
            "pooling_mode": pooling_mode,
            "residual": residual,
            "depth": len(layer_weights),
        },
    )



def graph_transformer_layer(
    features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    num_heads: int = 4,
    attention_alpha: float = 1.0,
    normalization: str = "layernorm",
    activation: str = "gelu",
    ff_gain: float = 2.0,
    residual: bool = True,
    norm_gamma: float = 1.0,
    norm_beta: float = 0.0,
    norm_eps: float = 1e-5,
) -> GraphTransformerResult:
    """Single simplified graph transformer block for multichannel node features."""
    x = _as_node_features(features)
    attn = multihead_graph_attention_multichannel(
        x,
        adjacency,
        num_heads=num_heads,
        alpha=attention_alpha,
        concat=False,
        activation="linear",
    )
    h = np.asarray(attn.output, dtype=np.float64)
    if residual and h.shape == x.shape:
        h = h + x
    h = np.asarray(graph_block_normalize_multichannel(h, mode=normalization, gamma=norm_gamma, beta=norm_beta, eps=norm_eps), dtype=np.float64)

    ff = _apply_activation(h, activation)
    ff = ff_gain * ff
    if residual and ff.shape == h.shape:
        ff = ff + h
    out = np.asarray(graph_block_normalize_multichannel(ff, mode=normalization, gamma=norm_gamma, beta=norm_beta, eps=norm_eps), dtype=np.float64)

    return GraphTransformerResult(
        output=out.tolist(),
        layers=[h.tolist(), out.tolist()],
        attention_matrices=attn.attention_matrices,
        meta={
            "num_heads": num_heads,
            "attention_alpha": attention_alpha,
            "normalization": normalization,
            "activation": activation,
            "ff_gain": ff_gain,
            "residual": residual,
            "channels": int(x.shape[1]),
        },
    )



def graph_transformer_stack(
    features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    depth: int = 2,
    num_heads: int = 4,
    attention_alpha: float = 1.0,
    normalization: str = "layernorm",
    activation: str = "gelu",
    ff_gain: float = 2.0,
    residual: bool = True,
    norm_gamma: float = 1.0,
    norm_beta: float = 0.0,
    norm_eps: float = 1e-5,
) -> GraphTransformerResult:
    """Stack several simplified graph transformer blocks."""
    depth = ensure_positive_int(depth, "depth")
    current = _as_node_features(features)
    layers: list[list[list[float]]] = []
    attention_matrices: list[list[list[float]]] = []
    for _ in range(depth):
        result = graph_transformer_layer(
            current,
            adjacency,
            num_heads=num_heads,
            attention_alpha=attention_alpha,
            normalization=normalization,
            activation=activation,
            ff_gain=ff_gain,
            residual=residual,
            norm_gamma=norm_gamma,
            norm_beta=norm_beta,
            norm_eps=norm_eps,
        )
        current = np.asarray(result.output, dtype=np.float64)
        layers.append(result.output)
        attention_matrices.append(result.attention_matrices[0])
    return GraphTransformerResult(
        output=current.tolist(),
        layers=layers,
        attention_matrices=attention_matrices,
        meta={
            "depth": depth,
            "num_heads": num_heads,
            "attention_alpha": attention_alpha,
            "normalization": normalization,
            "activation": activation,
            "ff_gain": ff_gain,
            "residual": residual,
            "channels": int(current.shape[1]),
        },
    )


@dataclass
class MessagePassingResult:
    output: list[list[float]]
    layers: list[list[list[float]]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QKVAttentionResult:
    output: list[list[float]]
    heads: list[list[list[float]]]
    attention_matrices: list[list[list[float]]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _edge_gate(edge_value: np.ndarray | float) -> float:
    arr = np.asarray(edge_value, dtype=np.float64)
    if arr.size == 0:
        return 1.0
    return 1.0 + float(np.mean(arr))



def edge_aware_message_passing(
    node_features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_features: list | np.ndarray,
    self_weight: float = 1.0,
    neighbor_weight: float = 1.0,
    aggregation: str = "mean",
    activation: str = "linear",
) -> list[list[float]]:
    """Message passing over node features using edge features as scalar gates."""
    x = _as_node_features(node_features)
    A = np.asarray(adjacency, dtype=np.float64)
    E = np.asarray(edge_features, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if x.shape[0] != A.shape[0]:
        raise ValueError("node count must match graph size")
    if E.shape[0] != A.shape[0] or E.shape[1] != A.shape[1]:
        raise ValueError("edge_features must match adjacency leading dimensions")

    out = np.zeros_like(x)
    for i in range(A.shape[0]):
        messages = []
        for j in range(A.shape[1]):
            if A[i, j] > 0:
                gate = _edge_gate(E[i, j])
                messages.append(neighbor_weight * gate * x[j])
        if messages:
            msg = np.stack(messages, axis=0)
            agg = np.mean(msg, axis=0) if aggregation == "mean" else np.sum(msg, axis=0)
        else:
            agg = np.zeros(x.shape[1], dtype=np.float64)
        out[i] = self_weight * x[i] + agg
    return _apply_activation(out, activation).tolist()



def edge_feature_message_passing_stack(
    node_features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_features: list | np.ndarray,
    depth: int = 2,
    self_weight: float = 1.0,
    neighbor_weight: float = 1.0,
    aggregation: str = "mean",
    activation: str = "relu",
    normalization: str = "layernorm",
    residual: bool = True,
) -> MessagePassingResult:
    depth = ensure_positive_int(depth, "depth")
    current = _as_node_features(node_features)
    layers: list[list[list[float]]] = []
    for _ in range(depth):
        out = np.asarray(
            edge_aware_message_passing(
                current,
                adjacency,
                edge_features,
                self_weight=self_weight,
                neighbor_weight=neighbor_weight,
                aggregation=aggregation,
                activation="linear",
            ),
            dtype=np.float64,
        )
        out = np.asarray(graph_block_normalize_multichannel(out, mode=normalization), dtype=np.float64)
        out = _apply_activation(out, activation)
        if residual and out.shape == current.shape:
            out = out + current
        layers.append(out.tolist())
        current = out
    return MessagePassingResult(output=current.tolist(), layers=layers, meta={"depth": depth, "aggregation": aggregation, "activation": activation, "normalization": normalization, "residual": residual})



def qkv_graph_attention(
    node_features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    num_heads: int = 4,
    query_scales: list[float] | None = None,
    key_scales: list[float] | None = None,
    value_scales: list[float] | None = None,
    concat: bool = False,
    add_self_loops: bool = True,
    activation: str = "linear",
    edge_features: list | np.ndarray | None = None,
) -> QKVAttentionResult:
    """Explicit Q/K/V attention for multichannel node features."""
    x = _as_node_features(node_features)
    A = np.asarray(adjacency, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if x.shape[0] != A.shape[0]:
        raise ValueError("node count must match graph size")
    num_heads = ensure_positive_int(num_heads, "num_heads")
    if add_self_loops:
        A = A + np.eye(A.shape[0], dtype=np.float64)

    if query_scales is None:
        query_scales = [1.0 + 0.05 * h for h in range(num_heads)]
    if key_scales is None:
        key_scales = [1.0 + 0.07 * h for h in range(num_heads)]
    if value_scales is None:
        value_scales = [1.0 + 0.09 * h for h in range(num_heads)]
    if not (len(query_scales) == len(key_scales) == len(value_scales) == num_heads):
        raise ValueError("projection scale lists must match num_heads")

    if edge_features is not None:
        E = np.asarray(edge_features, dtype=np.float64)
        if E.shape[0] != A.shape[0] or E.shape[1] != A.shape[1]:
            raise ValueError("edge_features must match adjacency leading dimensions")
    else:
        E = None

    head_outputs: list[list[list[float]]] = []
    head_matrices: list[list[list[float]]] = []
    d = float(np.sqrt(x.shape[1]))
    for h in range(num_heads):
        q = x * float(query_scales[h])
        k = x * float(key_scales[h])
        v = x * float(value_scales[h])
        scores = np.full((A.shape[0], A.shape[1]), -np.inf, dtype=np.float64)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] > 0:
                    bias = _edge_gate(E[i, j]) - 1.0 if E is not None else 0.0
                    scores[i, j] = float(np.dot(q[i], k[j]) / max(d, 1e-12) + bias)
            valid = np.isfinite(scores[i])
            row = scores[i, valid]
            row = row - np.max(row)
            exp_row = np.exp(row)
            scores[i, valid] = exp_row / max(np.sum(exp_row), 1e-12)
            scores[i, ~valid] = 0.0
        out = scores @ v
        out = _apply_activation(out, activation)
        head_outputs.append(out.tolist())
        head_matrices.append(scores.tolist())

    stacked = np.asarray(head_outputs, dtype=np.float64)
    output = np.concatenate([stacked[h] for h in range(num_heads)], axis=1) if concat else np.mean(stacked, axis=0)
    return QKVAttentionResult(output=output.tolist(), heads=head_outputs, attention_matrices=head_matrices, meta={"num_heads": num_heads, "concat": concat, "activation": activation, "channels_in": int(x.shape[1]), "channels_out": int(output.shape[1])})



def graph_transformer_qkv_layer(
    node_features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    num_heads: int = 4,
    query_scales: list[float] | None = None,
    key_scales: list[float] | None = None,
    value_scales: list[float] | None = None,
    normalization: str = "layernorm",
    activation: str = "gelu",
    ff_gain: float = 2.0,
    residual: bool = True,
    edge_features: list | np.ndarray | None = None,
) -> GraphTransformerResult:
    x = _as_node_features(node_features)
    attn = qkv_graph_attention(
        x,
        adjacency,
        num_heads=num_heads,
        query_scales=query_scales,
        key_scales=key_scales,
        value_scales=value_scales,
        concat=False,
        activation="linear",
        edge_features=edge_features,
    )
    h = np.asarray(attn.output, dtype=np.float64)
    if residual and h.shape == x.shape:
        h = h + x
    h = np.asarray(graph_block_normalize_multichannel(h, mode=normalization), dtype=np.float64)
    ff = _apply_activation(h, activation)
    ff = ff_gain * ff
    if residual and ff.shape == h.shape:
        ff = ff + h
    out = np.asarray(graph_block_normalize_multichannel(ff, mode=normalization), dtype=np.float64)
    return GraphTransformerResult(output=out.tolist(), layers=[h.tolist(), out.tolist()], attention_matrices=attn.attention_matrices, meta={"num_heads": num_heads, "normalization": normalization, "activation": activation, "ff_gain": ff_gain, "residual": residual, "channels": int(x.shape[1]), "explicit_qkv": True})



def graph_transformer_qkv_stack(
    node_features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    depth: int = 2,
    num_heads: int = 4,
    query_scales: list[float] | None = None,
    key_scales: list[float] | None = None,
    value_scales: list[float] | None = None,
    normalization: str = "layernorm",
    activation: str = "gelu",
    ff_gain: float = 2.0,
    residual: bool = True,
    edge_features: list | np.ndarray | None = None,
) -> GraphTransformerResult:
    depth = ensure_positive_int(depth, "depth")
    current = _as_node_features(node_features)
    layers: list[list[list[float]]] = []
    attention_matrices: list[list[list[float]]] = []
    for _ in range(depth):
        result = graph_transformer_qkv_layer(
            current,
            adjacency,
            num_heads=num_heads,
            query_scales=query_scales,
            key_scales=key_scales,
            value_scales=value_scales,
            normalization=normalization,
            activation=activation,
            ff_gain=ff_gain,
            residual=residual,
            edge_features=edge_features,
        )
        current = np.asarray(result.output, dtype=np.float64)
        layers.append(result.output)
        attention_matrices.append(result.attention_matrices[0])
    return GraphTransformerResult(output=current.tolist(), layers=layers, attention_matrices=attention_matrices, meta={"depth": depth, "num_heads": num_heads, "normalization": normalization, "activation": activation, "ff_gain": ff_gain, "residual": residual, "channels": int(current.shape[1]), "explicit_qkv": True})


@dataclass
class EdgeConditionedConvResult:
    output: list[list[float]]
    layers: list[list[list[float]]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def edge_conditioned_convolution(
    node_features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_features: list | np.ndarray,
    self_weight: float = 1.0,
    neighbor_weight: float = 1.0,
    aggregation: str = "mean",
    activation: str = "linear",
) -> list[list[float]]:
    """Edge-conditioned convolution using edge-driven channel gains."""
    x = _as_node_features(node_features)
    A = np.asarray(adjacency, dtype=np.float64)
    E = np.asarray(edge_features, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if x.shape[0] != A.shape[0]:
        raise ValueError("node count must match graph size")
    if E.shape[0] != A.shape[0] or E.shape[1] != A.shape[1]:
        raise ValueError("edge_features must match adjacency leading dimensions")

    out = np.zeros_like(x)
    for i in range(A.shape[0]):
        messages = []
        for j in range(A.shape[1]):
            if A[i, j] > 0:
                edge_vec = np.ravel(E[i, j]).astype(np.float64)
                if edge_vec.size == 0:
                    gain = np.ones(x.shape[1], dtype=np.float64)
                elif edge_vec.size == 1:
                    gain = np.ones(x.shape[1], dtype=np.float64) * (1.0 + float(edge_vec[0]))
                else:
                    repeats = int(np.ceil(x.shape[1] / edge_vec.size))
                    gain = 1.0 + np.tile(edge_vec, repeats)[: x.shape[1]]
                messages.append(neighbor_weight * gain * x[j])
        if messages:
            msg = np.stack(messages, axis=0)
            agg = np.mean(msg, axis=0) if aggregation == "mean" else np.sum(msg, axis=0)
        else:
            agg = np.zeros(x.shape[1], dtype=np.float64)
        out[i] = self_weight * x[i] + agg
    return _apply_activation(out, activation).tolist()



def edge_conditioned_conv_stack(
    node_features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_features: list | np.ndarray,
    depth: int = 2,
    self_weight: float = 1.0,
    neighbor_weight: float = 1.0,
    aggregation: str = "mean",
    activation: str = "relu",
    normalization: str = "layernorm",
    residual: bool = True,
) -> EdgeConditionedConvResult:
    depth = ensure_positive_int(depth, "depth")
    current = _as_node_features(node_features)
    layers: list[list[list[float]]] = []
    for _ in range(depth):
        out = np.asarray(
            edge_conditioned_convolution(
                current,
                adjacency,
                edge_features,
                self_weight=self_weight,
                neighbor_weight=neighbor_weight,
                aggregation=aggregation,
                activation="linear",
            ),
            dtype=np.float64,
        )
        out = np.asarray(graph_block_normalize_multichannel(out, mode=normalization), dtype=np.float64)
        out = _apply_activation(out, activation)
        if residual and out.shape == current.shape:
            out = out + current
        layers.append(out.tolist())
        current = out
    return EdgeConditionedConvResult(output=current.tolist(), layers=layers, meta={"depth": depth, "aggregation": aggregation, "activation": activation, "normalization": normalization, "residual": residual})



def masked_qkv_graph_attention(
    node_features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    num_heads: int = 4,
    query_scales: list[float] | None = None,
    key_scales: list[float] | None = None,
    value_scales: list[float] | None = None,
    concat: bool = False,
    add_self_loops: bool = True,
    activation: str = "linear",
    edge_features: list | np.ndarray | None = None,
    attention_mask: list[list[float]] | np.ndarray | None = None,
    edge_bias_scale: float = 1.0,
) -> QKVAttentionResult:
    """Q/K/V attention with optional explicit mask and richer edge-derived bias."""
    x = _as_node_features(node_features)
    A = np.asarray(adjacency, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if x.shape[0] != A.shape[0]:
        raise ValueError("node count must match graph size")
    num_heads = ensure_positive_int(num_heads, "num_heads")
    if add_self_loops:
        A = A + np.eye(A.shape[0], dtype=np.float64)
    if attention_mask is not None:
        M = np.asarray(attention_mask, dtype=np.float64)
        if M.shape != A.shape:
            raise ValueError("attention_mask must match adjacency shape")
        A = A * M
    if query_scales is None:
        query_scales = [1.0 + 0.05 * h for h in range(num_heads)]
    if key_scales is None:
        key_scales = [1.0 + 0.07 * h for h in range(num_heads)]
    if value_scales is None:
        value_scales = [1.0 + 0.09 * h for h in range(num_heads)]
    if not (len(query_scales) == len(key_scales) == len(value_scales) == num_heads):
        raise ValueError("projection scale lists must match num_heads")
    E = None
    if edge_features is not None:
        E = np.asarray(edge_features, dtype=np.float64)
        if E.shape[0] != A.shape[0] or E.shape[1] != A.shape[1]:
            raise ValueError("edge_features must match adjacency leading dimensions")

    head_outputs=[]
    head_matrices=[]
    d = float(np.sqrt(x.shape[1]))
    for h in range(num_heads):
        q = x * float(query_scales[h])
        k = x * float(key_scales[h])
        v = x * float(value_scales[h])
        probs = np.zeros((A.shape[0], A.shape[1]), dtype=np.float64)
        for i in range(A.shape[0]):
            valid = A[i] > 0
            if not np.any(valid):
                probs[i, i] = 1.0
                continue
            row_scores=[]
            idx=np.where(valid)[0]
            for j in idx:
                bias=0.0
                if E is not None:
                    edge_vec=np.ravel(E[i, j]).astype(np.float64)
                    bias = edge_bias_scale * (float(np.mean(edge_vec)) + float(np.std(edge_vec)))
                row_scores.append(float(np.dot(q[i], k[j]) / max(d, 1e-12) + bias))
            row_scores = np.asarray(row_scores, dtype=np.float64)
            row_scores = row_scores - np.max(row_scores)
            exp_scores = np.exp(row_scores)
            probs[i, idx] = exp_scores / max(np.sum(exp_scores), 1e-12)
        out = probs @ v
        out = _apply_activation(out, activation)
        head_outputs.append(out.tolist())
        head_matrices.append(probs.tolist())
    stacked=np.asarray(head_outputs, dtype=np.float64)
    output = np.concatenate([stacked[h] for h in range(num_heads)], axis=1) if concat else np.mean(stacked, axis=0)
    return QKVAttentionResult(output=output.tolist(), heads=head_outputs, attention_matrices=head_matrices, meta={"num_heads": num_heads, "concat": concat, "activation": activation, "channels_in": int(x.shape[1]), "channels_out": int(output.shape[1]), "masked": attention_mask is not None, "edge_bias_scale": edge_bias_scale})



def graph_transformer_masked_qkv_layer(
    node_features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    num_heads: int = 4,
    normalization: str = "layernorm",
    activation: str = "gelu",
    ff_gain: float = 2.0,
    residual: bool = True,
    edge_features: list | np.ndarray | None = None,
    attention_mask: list[list[float]] | np.ndarray | None = None,
    edge_bias_scale: float = 1.0,
) -> GraphTransformerResult:
    x = _as_node_features(node_features)
    attn = masked_qkv_graph_attention(x, adjacency, num_heads=num_heads, concat=False, activation="linear", edge_features=edge_features, attention_mask=attention_mask, edge_bias_scale=edge_bias_scale)
    h = np.asarray(attn.output, dtype=np.float64)
    if residual and h.shape == x.shape:
        h = h + x
    h = np.asarray(graph_block_normalize_multichannel(h, mode=normalization), dtype=np.float64)
    ff = _apply_activation(h, activation)
    ff = ff_gain * ff
    if residual and ff.shape == h.shape:
        ff = ff + h
    out = np.asarray(graph_block_normalize_multichannel(ff, mode=normalization), dtype=np.float64)
    return GraphTransformerResult(output=out.tolist(), layers=[h.tolist(), out.tolist()], attention_matrices=attn.attention_matrices, meta={"num_heads": num_heads, "normalization": normalization, "activation": activation, "ff_gain": ff_gain, "residual": residual, "channels": int(x.shape[1]), "explicit_qkv": True, "masked": attention_mask is not None, "edge_bias_scale": edge_bias_scale})



def graph_transformer_masked_qkv_stack(
    node_features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    depth: int = 2,
    num_heads: int = 4,
    normalization: str = "layernorm",
    activation: str = "gelu",
    ff_gain: float = 2.0,
    residual: bool = True,
    edge_features: list | np.ndarray | None = None,
    attention_mask: list[list[float]] | np.ndarray | None = None,
    edge_bias_scale: float = 1.0,
) -> GraphTransformerResult:
    depth = ensure_positive_int(depth, "depth")
    current = _as_node_features(node_features)
    layers=[]
    attention_matrices=[]
    for _ in range(depth):
        result = graph_transformer_masked_qkv_layer(current, adjacency, num_heads=num_heads, normalization=normalization, activation=activation, ff_gain=ff_gain, residual=residual, edge_features=edge_features, attention_mask=attention_mask, edge_bias_scale=edge_bias_scale)
        current = np.asarray(result.output, dtype=np.float64)
        layers.append(result.output)
        attention_matrices.append(result.attention_matrices[0])
    return GraphTransformerResult(output=current.tolist(), layers=layers, attention_matrices=attention_matrices, meta={"depth": depth, "num_heads": num_heads, "normalization": normalization, "activation": activation, "ff_gain": ff_gain, "residual": residual, "channels": int(current.shape[1]), "explicit_qkv": True, "masked": attention_mask is not None, "edge_bias_scale": edge_bias_scale})


@dataclass
class ChannelMixResult:
    output: list[list[float]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _auto_mix_matrix(in_channels: int, out_channels: int, mix_strength: float = 0.15) -> np.ndarray:
    base = np.zeros((out_channels, in_channels), dtype=np.float64)
    for i in range(out_channels):
        base[i, i % in_channels] = 1.0
    dense = np.ones((out_channels, in_channels), dtype=np.float64) / max(in_channels, 1)
    return (1.0 - mix_strength) * base + mix_strength * dense



def channel_mix(
    node_features: list[list[float]] | list[float] | np.ndarray,
    mix_matrix: list[list[float]] | np.ndarray | None = None,
    bias: list[float] | np.ndarray | None = None,
    activation: str = "linear",
    residual: bool = False,
    out_channels: int | None = None,
    mix_strength: float = 0.15,
) -> ChannelMixResult:
    """Apply learnable-style linear channel mixing to node features."""
    x = _as_node_features(node_features)
    in_channels = x.shape[1]
    if mix_matrix is None:
        out_channels = in_channels if out_channels is None else int(out_channels)
        W = _auto_mix_matrix(in_channels, out_channels, mix_strength=mix_strength)
    else:
        W = np.asarray(mix_matrix, dtype=np.float64)
        if W.ndim != 2 or W.shape[1] != in_channels:
            raise ValueError("mix_matrix must have shape (out_channels, in_channels)")
        out_channels = W.shape[0]
    b = np.zeros(out_channels, dtype=np.float64) if bias is None else np.asarray(bias, dtype=np.float64)
    if b.shape != (out_channels,):
        raise ValueError("bias must have shape (out_channels,)")
    y = x @ W.T + b
    y = _apply_activation(y, activation)
    if residual and y.shape == x.shape:
        y = y + x
    return ChannelMixResult(output=y.tolist(), meta={"in_channels": int(in_channels), "out_channels": int(out_channels), "activation": activation, "residual": residual, "auto": mix_matrix is None, "mix_strength": mix_strength})



def graph_transformer_enhanced_layer(
    node_features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    num_heads: int = 4,
    normalization: str = "layernorm",
    activation: str = "gelu",
    ff_gain: float = 2.0,
    residual: bool = True,
    edge_features: list | np.ndarray | None = None,
    attention_mask: list[list[float]] | np.ndarray | None = None,
    edge_bias_scale: float = 1.0,
    positional_encoding_method: str | None = None,
    positional_dimensions: int = 4,
    positional_steps: int = 4,
    channel_mix_matrix: list[list[float]] | np.ndarray | None = None,
    channel_mix_bias: list[float] | np.ndarray | None = None,
    mix_strength: float = 0.15,
) -> GraphTransformerResult:
    """Masked QKV transformer block with optional positional encodings and channel mixing."""
    x = _as_node_features(node_features)
    attn_input = x
    if positional_encoding_method is not None:
        attn_input = np.asarray(
            augment_with_graph_positional_encoding(
                x,
                adjacency,
                method=positional_encoding_method,
                dimensions=positional_dimensions,
                steps=positional_steps,
            ),
            dtype=np.float64,
        )
    attn = masked_qkv_graph_attention(
        attn_input,
        adjacency,
        num_heads=num_heads,
        concat=False,
        activation="linear",
        edge_features=edge_features,
        attention_mask=attention_mask,
        edge_bias_scale=edge_bias_scale,
    )
    mixed = channel_mix(
        attn.output,
        mix_matrix=channel_mix_matrix,
        bias=channel_mix_bias,
        activation="linear",
        residual=False,
        out_channels=x.shape[1],
        mix_strength=mix_strength,
    )
    h = np.asarray(mixed.output, dtype=np.float64)
    if residual and h.shape == x.shape:
        h = h + x
    h = np.asarray(graph_block_normalize_multichannel(h, mode=normalization), dtype=np.float64)
    ff = _apply_activation(h, activation)
    ff = ff_gain * ff
    if residual and ff.shape == h.shape:
        ff = ff + h
    out = np.asarray(graph_block_normalize_multichannel(ff, mode=normalization), dtype=np.float64)
    return GraphTransformerResult(output=out.tolist(), layers=[h.tolist(), out.tolist()], attention_matrices=attn.attention_matrices, meta={"num_heads": num_heads, "normalization": normalization, "activation": activation, "ff_gain": ff_gain, "residual": residual, "channels": int(x.shape[1]), "enhanced": True, "positional_encoding_method": positional_encoding_method, "edge_bias_scale": edge_bias_scale, "channel_mixing": mixed.meta})



def graph_transformer_enhanced_stack(
    node_features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    depth: int = 2,
    num_heads: int = 4,
    normalization: str = "layernorm",
    activation: str = "gelu",
    ff_gain: float = 2.0,
    residual: bool = True,
    edge_features: list | np.ndarray | None = None,
    attention_mask: list[list[float]] | np.ndarray | None = None,
    edge_bias_scale: float = 1.0,
    positional_encoding_method: str | None = None,
    positional_dimensions: int = 4,
    positional_steps: int = 4,
    channel_mix_matrix: list[list[float]] | np.ndarray | None = None,
    channel_mix_bias: list[float] | np.ndarray | None = None,
    mix_strength: float = 0.15,
) -> GraphTransformerResult:
    depth = ensure_positive_int(depth, "depth")
    current = _as_node_features(node_features)
    layers=[]
    attention_matrices=[]
    for _ in range(depth):
        result = graph_transformer_enhanced_layer(
            current,
            adjacency,
            num_heads=num_heads,
            normalization=normalization,
            activation=activation,
            ff_gain=ff_gain,
            residual=residual,
            edge_features=edge_features,
            attention_mask=attention_mask,
            edge_bias_scale=edge_bias_scale,
            positional_encoding_method=positional_encoding_method,
            positional_dimensions=positional_dimensions,
            positional_steps=positional_steps,
            channel_mix_matrix=channel_mix_matrix,
            channel_mix_bias=channel_mix_bias,
            mix_strength=mix_strength,
        )
        current = np.asarray(result.output, dtype=np.float64)
        layers.append(result.output)
        attention_matrices.append(result.attention_matrices[0])
    return GraphTransformerResult(output=current.tolist(), layers=layers, attention_matrices=attention_matrices, meta={"depth": depth, "num_heads": num_heads, "normalization": normalization, "activation": activation, "ff_gain": ff_gain, "residual": residual, "channels": int(current.shape[1]), "enhanced": True, "positional_encoding_method": positional_encoding_method, "edge_bias_scale": edge_bias_scale, "mix_strength": mix_strength})



def structured_edge_embedding_attention(
    node_features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_embeddings: list | np.ndarray,
    num_heads: int = 4,
    concat: bool = False,
    activation: str = "linear",
    add_self_loops: bool = True,
    edge_embedding_scale: float = 1.0,
) -> QKVAttentionResult:
    """Attention using structured edge embeddings as richer bias terms."""
    x = _as_node_features(node_features)
    A = np.asarray(adjacency, dtype=np.float64)
    E = np.asarray(edge_embeddings, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if x.shape[0] != A.shape[0]:
        raise ValueError("node count must match graph size")
    if E.shape[0] != A.shape[0] or E.shape[1] != A.shape[1]:
        raise ValueError("edge_embeddings must match adjacency leading dimensions")
    if add_self_loops:
        A = A + np.eye(A.shape[0], dtype=np.float64)
    num_heads = ensure_positive_int(num_heads, "num_heads")

    head_outputs=[]
    head_matrices=[]
    d = float(np.sqrt(x.shape[1]))
    for h in range(num_heads):
        q = x * (1.0 + 0.05 * h)
        k = x * (1.0 + 0.07 * h)
        v = x * (1.0 + 0.09 * h)
        probs = np.zeros((A.shape[0], A.shape[1]), dtype=np.float64)
        for i in range(A.shape[0]):
            valid = A[i] > 0
            if not np.any(valid):
                probs[i, i] = 1.0
                continue
            idx = np.where(valid)[0]
            scores=[]
            for j in idx:
                edge_vec = np.ravel(E[i, j]).astype(np.float64)
                emb_bias = edge_embedding_scale * (float(np.mean(edge_vec)) + 0.5 * float(np.std(edge_vec)) + 0.25 * float(np.max(edge_vec)) - 0.25 * float(np.min(edge_vec)))
                scores.append(float(np.dot(q[i], k[j]) / max(d, 1e-12) + emb_bias))
            scores = np.asarray(scores, dtype=np.float64)
            scores = scores - np.max(scores)
            exp_scores = np.exp(scores)
            probs[i, idx] = exp_scores / max(np.sum(exp_scores), 1e-12)
        out = probs @ v
        out = _apply_activation(out, activation)
        head_outputs.append(out.tolist())
        head_matrices.append(probs.tolist())
    stacked = np.asarray(head_outputs, dtype=np.float64)
    output = np.concatenate([stacked[h] for h in range(num_heads)], axis=1) if concat else np.mean(stacked, axis=0)
    return QKVAttentionResult(output=output.tolist(), heads=head_outputs, attention_matrices=head_matrices, meta={"num_heads": num_heads, "concat": concat, "activation": activation, "channels_in": int(x.shape[1]), "channels_out": int(output.shape[1]), "structured_edge_embeddings": True, "edge_embedding_scale": edge_embedding_scale})



def graph_transformer_edge_embedding_layer(
    node_features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_embeddings: list | np.ndarray,
    num_heads: int = 4,
    normalization: str = "layernorm",
    activation: str = "gelu",
    ff_gain: float = 2.0,
    residual: bool = True,
    edge_embedding_scale: float = 1.0,
) -> GraphTransformerResult:
    x = _as_node_features(node_features)
    attn = structured_edge_embedding_attention(
        x,
        adjacency,
        edge_embeddings,
        num_heads=num_heads,
        concat=False,
        activation="linear",
        edge_embedding_scale=edge_embedding_scale,
    )
    h = np.asarray(attn.output, dtype=np.float64)
    if residual and h.shape == x.shape:
        h = h + x
    h = np.asarray(graph_block_normalize_multichannel(h, mode=normalization), dtype=np.float64)
    ff = _apply_activation(h, activation)
    ff = ff_gain * ff
    if residual and ff.shape == h.shape:
        ff = ff + h
    out = np.asarray(graph_block_normalize_multichannel(ff, mode=normalization), dtype=np.float64)
    return GraphTransformerResult(output=out.tolist(), layers=[h.tolist(), out.tolist()], attention_matrices=attn.attention_matrices, meta={"num_heads": num_heads, "normalization": normalization, "activation": activation, "ff_gain": ff_gain, "residual": residual, "channels": int(x.shape[1]), "structured_edge_embeddings": True, "edge_embedding_scale": edge_embedding_scale})



def graph_transformer_edge_embedding_stack(
    node_features: list[list[float]] | list[float] | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_embeddings: list | np.ndarray,
    depth: int = 2,
    num_heads: int = 4,
    normalization: str = "layernorm",
    activation: str = "gelu",
    ff_gain: float = 2.0,
    residual: bool = True,
    edge_embedding_scale: float = 1.0,
) -> GraphTransformerResult:
    depth = ensure_positive_int(depth, "depth")
    current = _as_node_features(node_features)
    layers=[]
    attention_matrices=[]
    for _ in range(depth):
        result = graph_transformer_edge_embedding_layer(
            current,
            adjacency,
            edge_embeddings,
            num_heads=num_heads,
            normalization=normalization,
            activation=activation,
            ff_gain=ff_gain,
            residual=residual,
            edge_embedding_scale=edge_embedding_scale,
        )
        current = np.asarray(result.output, dtype=np.float64)
        layers.append(result.output)
        attention_matrices.append(result.attention_matrices[0])
    return GraphTransformerResult(output=current.tolist(), layers=layers, attention_matrices=attention_matrices, meta={"depth": depth, "num_heads": num_heads, "normalization": normalization, "activation": activation, "ff_gain": ff_gain, "residual": residual, "channels": int(current.shape[1]), "structured_edge_embeddings": True, "edge_embedding_scale": edge_embedding_scale})


@dataclass
class HybridTemporalAttentionResult:
    output: list
    spatial_attention_matrices: list[list[list[float]]]
    temporal_weights: list[list[float]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def hybrid_node_edge_temporal_attention(
    sequence: list | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_embeddings: list | np.ndarray,
    num_heads: int = 4,
    temporal_window: int = 1,
    activation: str = "linear",
    edge_embedding_scale: float = 1.0,
    temporal_decay: float = 1.0,
) -> HybridTemporalAttentionResult:
    """Apply spatial node-edge attention and temporal aggregation on a sequence of graph signals."""
    x = np.asarray(sequence, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("sequence must have shape (time, nodes, channels)")
    temporal_window = ensure_positive_int(temporal_window, "temporal_window")
    spatial_outputs: list[np.ndarray] = []
    spatial_mats: list[list[list[float]]] = []
    for t in range(x.shape[0]):
        attn = structured_edge_embedding_attention(
            x[t], adjacency, edge_embeddings, num_heads=num_heads, concat=False, activation=activation, edge_embedding_scale=edge_embedding_scale
        )
        spatial_outputs.append(np.asarray(attn.output, dtype=np.float64))
        spatial_mats.append(attn.attention_matrices[0])

    spatial = np.stack(spatial_outputs, axis=0)
    outputs = np.zeros_like(spatial)
    temporal_weights: list[list[float]] = []
    for t in range(spatial.shape[0]):
        idxs=[]
        weights=[]
        for tau in range(max(0, t - temporal_window), min(spatial.shape[0], t + temporal_window + 1)):
            idxs.append(tau)
            weights.append(float(np.exp(-temporal_decay * abs(t - tau))))
        w = np.asarray(weights, dtype=np.float64)
        w = w / np.sum(w)
        temporal_weights.append(w.tolist())
        acc = np.zeros_like(spatial[t])
        for wt, tau in zip(w, idxs):
            acc += wt * spatial[tau]
        outputs[t] = acc
    return HybridTemporalAttentionResult(output=outputs.tolist(), spatial_attention_matrices=spatial_mats, temporal_weights=temporal_weights, meta={"time_steps": int(x.shape[0]), "nodes": int(x.shape[1]), "channels": int(x.shape[2]), "num_heads": num_heads, "temporal_window": temporal_window, "edge_embedding_scale": edge_embedding_scale, "temporal_decay": temporal_decay})



def hybrid_graph_temporal_transformer_layer(
    sequence: list | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_embeddings: list | np.ndarray,
    num_heads: int = 4,
    temporal_window: int = 1,
    normalization: str = "layernorm",
    activation: str = "gelu",
    ff_gain: float = 2.0,
    residual: bool = True,
    edge_embedding_scale: float = 1.0,
    temporal_decay: float = 1.0,
) -> HybridTemporalAttentionResult:
    x = np.asarray(sequence, dtype=np.float64)
    base = hybrid_node_edge_temporal_attention(
        x,
        adjacency,
        edge_embeddings,
        num_heads=num_heads,
        temporal_window=temporal_window,
        activation="linear",
        edge_embedding_scale=edge_embedding_scale,
        temporal_decay=temporal_decay,
    )
    y = np.asarray(base.output, dtype=np.float64)
    if residual and y.shape == x.shape:
        y = y + x
    z = []
    for t in range(y.shape[0]):
        block = np.asarray(graph_block_normalize_multichannel(y[t], mode=normalization), dtype=np.float64)
        ff = ff_gain * _apply_activation(block, activation)
        if residual and ff.shape == block.shape:
            ff = ff + block
        z.append(np.asarray(graph_block_normalize_multichannel(ff, mode=normalization), dtype=np.float64))
    out = np.stack(z, axis=0)
    return HybridTemporalAttentionResult(output=out.tolist(), spatial_attention_matrices=base.spatial_attention_matrices, temporal_weights=base.temporal_weights, meta={"num_heads": num_heads, "temporal_window": temporal_window, "normalization": normalization, "activation": activation, "ff_gain": ff_gain, "residual": residual, "edge_embedding_scale": edge_embedding_scale, "temporal_decay": temporal_decay, "hybrid": True})



def hybrid_graph_temporal_transformer_stack(
    sequence: list | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_embeddings: list | np.ndarray,
    depth: int = 2,
    num_heads: int = 4,
    temporal_window: int = 1,
    normalization: str = "layernorm",
    activation: str = "gelu",
    ff_gain: float = 2.0,
    residual: bool = True,
    edge_embedding_scale: float = 1.0,
    temporal_decay: float = 1.0,
) -> HybridTemporalAttentionResult:
    depth = ensure_positive_int(depth, "depth")
    current = np.asarray(sequence, dtype=np.float64)
    last: HybridTemporalAttentionResult | None = None
    for _ in range(depth):
        last = hybrid_graph_temporal_transformer_layer(
            current,
            adjacency,
            edge_embeddings,
            num_heads=num_heads,
            temporal_window=temporal_window,
            normalization=normalization,
            activation=activation,
            ff_gain=ff_gain,
            residual=residual,
            edge_embedding_scale=edge_embedding_scale,
            temporal_decay=temporal_decay,
        )
        current = np.asarray(last.output, dtype=np.float64)
    assert last is not None
    last.meta["depth"] = depth
    return last


@dataclass
class RecurrentHybridAttentionResult:
    output: list
    memory_states: list
    spatial_attention_matrices: list[list[list[float]]]
    temporal_weights: list[list[float]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def recurrent_hybrid_node_edge_temporal_attention(
    sequence: list | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_embeddings: list | np.ndarray,
    num_heads: int = 4,
    temporal_window: int = 1,
    activation: str = "linear",
    edge_embedding_scale: float = 1.0,
    temporal_decay: float = 1.0,
    memory_decay: float = 0.8,
) -> RecurrentHybridAttentionResult:
    """Hybrid node-edge-temporal attention with recurrent memory accumulation."""
    base = hybrid_node_edge_temporal_attention(
        sequence,
        adjacency,
        edge_embeddings,
        num_heads=num_heads,
        temporal_window=temporal_window,
        activation=activation,
        edge_embedding_scale=edge_embedding_scale,
        temporal_decay=temporal_decay,
    )
    x = np.asarray(base.output, dtype=np.float64)
    memory = np.zeros_like(x[0])
    outputs = []
    memories = []
    for t in range(x.shape[0]):
        memory = memory_decay * memory + (1.0 - memory_decay) * x[t]
        out = 0.5 * (x[t] + memory)
        outputs.append(out.tolist())
        memories.append(memory.tolist())
    return RecurrentHybridAttentionResult(
        output=outputs,
        memory_states=memories,
        spatial_attention_matrices=base.spatial_attention_matrices,
        temporal_weights=base.temporal_weights,
        meta={
            "time_steps": int(x.shape[0]),
            "nodes": int(x.shape[1]),
            "channels": int(x.shape[2]),
            "num_heads": num_heads,
            "temporal_window": temporal_window,
            "edge_embedding_scale": edge_embedding_scale,
            "temporal_decay": temporal_decay,
            "memory_decay": memory_decay,
            "recurrent": True,
        },
    )



def recurrent_hybrid_graph_temporal_transformer_stack(
    sequence: list | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_embeddings: list | np.ndarray,
    depth: int = 2,
    num_heads: int = 4,
    temporal_window: int = 1,
    normalization: str = "layernorm",
    activation: str = "gelu",
    ff_gain: float = 2.0,
    residual: bool = True,
    edge_embedding_scale: float = 1.0,
    temporal_decay: float = 1.0,
    memory_decay: float = 0.8,
) -> RecurrentHybridAttentionResult:
    depth = ensure_positive_int(depth, "depth")
    current = np.asarray(sequence, dtype=np.float64)
    last: RecurrentHybridAttentionResult | None = None
    for _ in range(depth):
        base = recurrent_hybrid_node_edge_temporal_attention(
            current,
            adjacency,
            edge_embeddings,
            num_heads=num_heads,
            temporal_window=temporal_window,
            activation="linear",
            edge_embedding_scale=edge_embedding_scale,
            temporal_decay=temporal_decay,
            memory_decay=memory_decay,
        )
        y = np.asarray(base.output, dtype=np.float64)
        z=[]
        for t in range(y.shape[0]):
            block = np.asarray(graph_block_normalize_multichannel(y[t], mode=normalization), dtype=np.float64)
            ff = ff_gain * _apply_activation(block, activation)
            if residual and ff.shape == block.shape:
                ff = ff + block
            z.append(np.asarray(graph_block_normalize_multichannel(ff, mode=normalization), dtype=np.float64))
        current = np.stack(z, axis=0)
        last = RecurrentHybridAttentionResult(
            output=current.tolist(),
            memory_states=base.memory_states,
            spatial_attention_matrices=base.spatial_attention_matrices,
            temporal_weights=base.temporal_weights,
            meta={
                "num_heads": num_heads,
                "temporal_window": temporal_window,
                "normalization": normalization,
                "activation": activation,
                "ff_gain": ff_gain,
                "residual": residual,
                "edge_embedding_scale": edge_embedding_scale,
                "temporal_decay": temporal_decay,
                "memory_decay": memory_decay,
                "recurrent": True,
            },
        )
    assert last is not None
    last.meta["depth"] = depth
    return last


@dataclass
class HybridGatedMemoryResult:
    output: list
    hidden_states: list
    cell_states: list | None
    spatial_attention_matrices: list[list[list[float]]]
    temporal_weights: list[list[float]]
    gates: dict[str, list]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))



def hybrid_graph_temporal_gated_memory(
    sequence: list | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_embeddings: list | np.ndarray,
    num_heads: int = 4,
    temporal_window: int = 1,
    mode: str = "gru",
    gate_bias: float = 0.0,
    memory_decay: float = 0.8,
    edge_embedding_scale: float = 1.0,
    temporal_decay: float = 1.0,
) -> HybridGatedMemoryResult:
    """Apply hybrid node-edge-temporal attention followed by gated recurrent memory."""
    base = hybrid_node_edge_temporal_attention(
        sequence,
        adjacency,
        edge_embeddings,
        num_heads=num_heads,
        temporal_window=temporal_window,
        activation="linear",
        edge_embedding_scale=edge_embedding_scale,
        temporal_decay=temporal_decay,
    )
    x = np.asarray(base.output, dtype=np.float64)
    h = np.zeros_like(x[0])
    c = np.zeros_like(x[0])
    outputs = []
    hidden_states = []
    cell_states = []
    gates: dict[str, list] = {"update": [], "reset": [], "forget": [], "input": [], "output": []}
    for t in range(x.shape[0]):
        xt = x[t]
        if mode == "gru":
            z = _sigmoid(xt + memory_decay * h + gate_bias)
            r = _sigmoid(xt - memory_decay * h - gate_bias)
            candidate = np.tanh(xt + r * h)
            h = (1.0 - z) * h + z * candidate
            outputs.append(h.tolist())
            hidden_states.append(h.tolist())
            cell_states.append((np.zeros_like(h)).tolist())
            gates["update"].append(z.tolist())
            gates["reset"].append(r.tolist())
        elif mode == "lstm":
            f = _sigmoid(xt + memory_decay * h + gate_bias)
            i = _sigmoid(xt - memory_decay * h - gate_bias)
            o = _sigmoid(xt + h)
            g = np.tanh(xt + 0.5 * h)
            c = f * c + i * g
            h = o * np.tanh(c)
            outputs.append(h.tolist())
            hidden_states.append(h.tolist())
            cell_states.append(c.tolist())
            gates["forget"].append(f.tolist())
            gates["input"].append(i.tolist())
            gates["output"].append(o.tolist())
        else:
            raise ValueError("mode must be 'gru' or 'lstm'")
    return HybridGatedMemoryResult(
        output=outputs,
        hidden_states=hidden_states,
        cell_states=None if mode == "gru" else cell_states,
        spatial_attention_matrices=base.spatial_attention_matrices,
        temporal_weights=base.temporal_weights,
        gates=gates,
        meta={
            "time_steps": int(x.shape[0]),
            "nodes": int(x.shape[1]),
            "channels": int(x.shape[2]),
            "num_heads": num_heads,
            "temporal_window": temporal_window,
            "mode": mode,
            "gate_bias": gate_bias,
            "memory_decay": memory_decay,
            "edge_embedding_scale": edge_embedding_scale,
            "temporal_decay": temporal_decay,
        },
    )



def hybrid_graph_temporal_gated_stack(
    sequence: list | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_embeddings: list | np.ndarray,
    depth: int = 2,
    num_heads: int = 4,
    temporal_window: int = 1,
    mode: str = "gru",
    normalization: str = "layernorm",
    activation: str = "gelu",
    residual: bool = True,
    gate_bias: float = 0.0,
    memory_decay: float = 0.8,
    edge_embedding_scale: float = 1.0,
    temporal_decay: float = 1.0,
) -> HybridGatedMemoryResult:
    """Stack hybrid attention blocks with simplified GRU/LSTM-style memory."""
    depth = ensure_positive_int(depth, "depth")
    current = np.asarray(sequence, dtype=np.float64)
    last: HybridGatedMemoryResult | None = None
    for _ in range(depth):
        base = hybrid_graph_temporal_gated_memory(
            current,
            adjacency,
            edge_embeddings,
            num_heads=num_heads,
            temporal_window=temporal_window,
            mode=mode,
            gate_bias=gate_bias,
            memory_decay=memory_decay,
            edge_embedding_scale=edge_embedding_scale,
            temporal_decay=temporal_decay,
        )
        y = np.asarray(base.output, dtype=np.float64)
        z = []
        for t in range(y.shape[0]):
            block = np.asarray(graph_block_normalize_multichannel(y[t], mode=normalization), dtype=np.float64)
            ff = _apply_activation(block, activation)
            if residual and ff.shape == block.shape:
                ff = ff + block
            z.append(np.asarray(graph_block_normalize_multichannel(ff, mode=normalization), dtype=np.float64))
        current = np.stack(z, axis=0)
        last = HybridGatedMemoryResult(
            output=current.tolist(),
            hidden_states=base.hidden_states,
            cell_states=base.cell_states,
            spatial_attention_matrices=base.spatial_attention_matrices,
            temporal_weights=base.temporal_weights,
            gates=base.gates,
            meta={
                "depth": depth,
                "num_heads": num_heads,
                "temporal_window": temporal_window,
                "mode": mode,
                "normalization": normalization,
                "activation": activation,
                "residual": residual,
                "gate_bias": gate_bias,
                "memory_decay": memory_decay,
                "edge_embedding_scale": edge_embedding_scale,
                "temporal_decay": temporal_decay,
            },
        )
    assert last is not None
    return last


@dataclass
class BidirectionalGatedMemoryResult:
    output: list
    forward_states: list
    backward_states: list
    cell_states: dict[str, list | None]
    spatial_attention_matrices: list[list[list[float]]]
    temporal_weights: list[list[float]]
    gates: dict[str, dict[str, list]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def bidirectional_hybrid_graph_temporal_gated_memory(
    sequence: list | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_embeddings: list | np.ndarray,
    num_heads: int = 4,
    temporal_window: int = 1,
    mode: str = "gru",
    merge: str = "mean",
    gate_bias: float = 0.0,
    memory_decay: float = 0.8,
    edge_embedding_scale: float = 1.0,
    temporal_decay: float = 1.0,
) -> BidirectionalGatedMemoryResult:
    """Run simplified gated memory in forward and backward temporal directions."""
    seq = np.asarray(sequence, dtype=np.float64)
    forward = hybrid_graph_temporal_gated_memory(
        seq,
        adjacency,
        edge_embeddings,
        num_heads=num_heads,
        temporal_window=temporal_window,
        mode=mode,
        gate_bias=gate_bias,
        memory_decay=memory_decay,
        edge_embedding_scale=edge_embedding_scale,
        temporal_decay=temporal_decay,
    )
    backward = hybrid_graph_temporal_gated_memory(
        seq[::-1],
        adjacency,
        edge_embeddings,
        num_heads=num_heads,
        temporal_window=temporal_window,
        mode=mode,
        gate_bias=gate_bias,
        memory_decay=memory_decay,
        edge_embedding_scale=edge_embedding_scale,
        temporal_decay=temporal_decay,
    )
    f = np.asarray(forward.output, dtype=np.float64)
    b = np.asarray(backward.output, dtype=np.float64)[::-1]
    if merge == "mean":
        out = 0.5 * (f + b)
    elif merge == "concat":
        out = np.concatenate([f, b], axis=2)
    else:
        raise ValueError("merge must be 'mean' or 'concat'")
    backward_hidden = list(reversed(backward.hidden_states))
    backward_cell = None if backward.cell_states is None else list(reversed(backward.cell_states))
    return BidirectionalGatedMemoryResult(
        output=out.tolist(),
        forward_states=forward.hidden_states,
        backward_states=backward_hidden,
        cell_states={"forward": forward.cell_states, "backward": backward_cell},
        spatial_attention_matrices=forward.spatial_attention_matrices,
        temporal_weights=forward.temporal_weights,
        gates={"forward": forward.gates, "backward": backward.gates},
        meta={
            "time_steps": int(seq.shape[0]),
            "nodes": int(seq.shape[1]),
            "channels": int(out.shape[2]),
            "num_heads": num_heads,
            "temporal_window": temporal_window,
            "mode": mode,
            "merge": merge,
            "gate_bias": gate_bias,
            "memory_decay": memory_decay,
            "edge_embedding_scale": edge_embedding_scale,
            "temporal_decay": temporal_decay,
            "bidirectional": True,
        },
    )



def bidirectional_hybrid_graph_temporal_gated_stack(
    sequence: list | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_embeddings: list | np.ndarray,
    depth: int = 2,
    num_heads: int = 4,
    temporal_window: int = 1,
    mode: str = "gru",
    merge: str = "mean",
    normalization: str = "layernorm",
    activation: str = "gelu",
    residual: bool = True,
    gate_bias: float = 0.0,
    memory_decay: float = 0.8,
    edge_embedding_scale: float = 1.0,
    temporal_decay: float = 1.0,
) -> BidirectionalGatedMemoryResult:
    """Stack bidirectional gated hybrid memory blocks."""
    depth = ensure_positive_int(depth, "depth")
    current = np.asarray(sequence, dtype=np.float64)
    last: BidirectionalGatedMemoryResult | None = None
    for _ in range(depth):
        base = bidirectional_hybrid_graph_temporal_gated_memory(
            current,
            adjacency,
            edge_embeddings,
            num_heads=num_heads,
            temporal_window=temporal_window,
            mode=mode,
            merge=merge,
            gate_bias=gate_bias,
            memory_decay=memory_decay,
            edge_embedding_scale=edge_embedding_scale,
            temporal_decay=temporal_decay,
        )
        y = np.asarray(base.output, dtype=np.float64)
        z = []
        for t in range(y.shape[0]):
            block = np.asarray(graph_block_normalize_multichannel(y[t], mode=normalization), dtype=np.float64)
            ff = _apply_activation(block, activation)
            if residual and ff.shape == block.shape:
                ff = ff + block
            z.append(np.asarray(graph_block_normalize_multichannel(ff, mode=normalization), dtype=np.float64))
        current = np.stack(z, axis=0)
        last = BidirectionalGatedMemoryResult(
            output=current.tolist(),
            forward_states=base.forward_states,
            backward_states=base.backward_states,
            cell_states=base.cell_states,
            spatial_attention_matrices=base.spatial_attention_matrices,
            temporal_weights=base.temporal_weights,
            gates=base.gates,
            meta={
                "depth": depth,
                "num_heads": num_heads,
                "temporal_window": temporal_window,
                "mode": mode,
                "merge": merge,
                "normalization": normalization,
                "activation": activation,
                "residual": residual,
                "gate_bias": gate_bias,
                "memory_decay": memory_decay,
                "edge_embedding_scale": edge_embedding_scale,
                "temporal_decay": temporal_decay,
                "bidirectional": True,
            },
        )
    assert last is not None
    return last


@dataclass
class HierarchicalGatedMemoryResult:
    output: list
    fast_states: list
    slow_states: list
    cell_states: dict[str, list | None]
    spatial_attention_matrices: list[list[list[float]]]
    temporal_weights: list[list[float]]
    gates: dict[str, list]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def hierarchical_hybrid_graph_temporal_gated_memory(
    sequence: list | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_embeddings: list | np.ndarray,
    num_heads: int = 4,
    temporal_window: int = 1,
    mode: str = "gru",
    gate_bias: float = 0.0,
    memory_decay: float = 0.8,
    slow_decay: float = 0.95,
    edge_embedding_scale: float = 1.0,
    temporal_decay: float = 1.0,
) -> HierarchicalGatedMemoryResult:
    """Hybrid attention with hierarchical fast/slow gated states."""
    base = hybrid_node_edge_temporal_attention(
        sequence,
        adjacency,
        edge_embeddings,
        num_heads=num_heads,
        temporal_window=temporal_window,
        activation="linear",
        edge_embedding_scale=edge_embedding_scale,
        temporal_decay=temporal_decay,
    )
    x = np.asarray(base.output, dtype=np.float64)
    h_fast = np.zeros_like(x[0])
    h_slow = np.zeros_like(x[0])
    c_fast = np.zeros_like(x[0])
    c_slow = np.zeros_like(x[0])
    outputs=[]
    fast_states=[]
    slow_states=[]
    gates={"fast": [], "slow": []}
    cell_fast=[]
    cell_slow=[]
    for t in range(x.shape[0]):
        xt=x[t]
        if mode == "gru":
            z_fast = _sigmoid(xt + memory_decay * h_fast + gate_bias)
            cand_fast = np.tanh(xt + h_fast)
            h_fast = (1.0 - z_fast) * h_fast + z_fast * cand_fast
            z_slow = _sigmoid(h_fast + slow_decay * h_slow + 0.5 * gate_bias)
            cand_slow = np.tanh(h_fast + h_slow)
            h_slow = (1.0 - z_slow) * h_slow + z_slow * cand_slow
            out = 0.5 * (h_fast + h_slow)
            gates["fast"].append(z_fast.tolist())
            gates["slow"].append(z_slow.tolist())
            cell_fast.append((np.zeros_like(h_fast)).tolist())
            cell_slow.append((np.zeros_like(h_slow)).tolist())
        elif mode == "lstm":
            f_fast = _sigmoid(xt + h_fast + gate_bias)
            i_fast = _sigmoid(xt - h_fast - gate_bias)
            g_fast = np.tanh(xt + 0.5 * h_fast)
            c_fast = f_fast * c_fast + i_fast * g_fast
            h_fast = np.tanh(c_fast)
            f_slow = _sigmoid(h_fast + slow_decay * h_slow + 0.5 * gate_bias)
            i_slow = _sigmoid(h_fast - h_slow - 0.5 * gate_bias)
            g_slow = np.tanh(h_fast + 0.5 * h_slow)
            c_slow = f_slow * c_slow + i_slow * g_slow
            h_slow = np.tanh(c_slow)
            out = 0.5 * (h_fast + h_slow)
            gates["fast"].append({"forget": f_fast.tolist(), "input": i_fast.tolist()})
            gates["slow"].append({"forget": f_slow.tolist(), "input": i_slow.tolist()})
            cell_fast.append(c_fast.tolist())
            cell_slow.append(c_slow.tolist())
        else:
            raise ValueError("mode must be 'gru' or 'lstm'")
        outputs.append(out.tolist())
        fast_states.append(h_fast.tolist())
        slow_states.append(h_slow.tolist())
    return HierarchicalGatedMemoryResult(
        output=outputs,
        fast_states=fast_states,
        slow_states=slow_states,
        cell_states={"fast": None if mode == "gru" else cell_fast, "slow": None if mode == "gru" else cell_slow},
        spatial_attention_matrices=base.spatial_attention_matrices,
        temporal_weights=base.temporal_weights,
        gates=gates,
        meta={
            "time_steps": int(x.shape[0]),
            "nodes": int(x.shape[1]),
            "channels": int(x.shape[2]),
            "num_heads": num_heads,
            "temporal_window": temporal_window,
            "mode": mode,
            "gate_bias": gate_bias,
            "memory_decay": memory_decay,
            "slow_decay": slow_decay,
            "edge_embedding_scale": edge_embedding_scale,
            "temporal_decay": temporal_decay,
            "hierarchical": True,
        },
    )



def hierarchical_hybrid_graph_temporal_gated_stack(
    sequence: list | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_embeddings: list | np.ndarray,
    depth: int = 2,
    num_heads: int = 4,
    temporal_window: int = 1,
    mode: str = "gru",
    normalization: str = "layernorm",
    activation: str = "gelu",
    residual: bool = True,
    gate_bias: float = 0.0,
    memory_decay: float = 0.8,
    slow_decay: float = 0.95,
    edge_embedding_scale: float = 1.0,
    temporal_decay: float = 1.0,
) -> HierarchicalGatedMemoryResult:
    """Stack hierarchical gated memory blocks on top of hybrid temporal attention."""
    depth = ensure_positive_int(depth, "depth")
    current = np.asarray(sequence, dtype=np.float64)
    last: HierarchicalGatedMemoryResult | None = None
    for _ in range(depth):
        base = hierarchical_hybrid_graph_temporal_gated_memory(
            current,
            adjacency,
            edge_embeddings,
            num_heads=num_heads,
            temporal_window=temporal_window,
            mode=mode,
            gate_bias=gate_bias,
            memory_decay=memory_decay,
            slow_decay=slow_decay,
            edge_embedding_scale=edge_embedding_scale,
            temporal_decay=temporal_decay,
        )
        y = np.asarray(base.output, dtype=np.float64)
        z=[]
        for t in range(y.shape[0]):
            block = np.asarray(graph_block_normalize_multichannel(y[t], mode=normalization), dtype=np.float64)
            ff = _apply_activation(block, activation)
            if residual and ff.shape == block.shape:
                ff = ff + block
            z.append(np.asarray(graph_block_normalize_multichannel(ff, mode=normalization), dtype=np.float64))
        current = np.stack(z, axis=0)
        last = HierarchicalGatedMemoryResult(
            output=current.tolist(),
            fast_states=base.fast_states,
            slow_states=base.slow_states,
            cell_states=base.cell_states,
            spatial_attention_matrices=base.spatial_attention_matrices,
            temporal_weights=base.temporal_weights,
            gates=base.gates,
            meta={
                "depth": depth,
                "num_heads": num_heads,
                "temporal_window": temporal_window,
                "mode": mode,
                "normalization": normalization,
                "activation": activation,
                "residual": residual,
                "gate_bias": gate_bias,
                "memory_decay": memory_decay,
                "slow_decay": slow_decay,
                "edge_embedding_scale": edge_embedding_scale,
                "temporal_decay": temporal_decay,
                "hierarchical": True,
            },
        )
    assert last is not None
    return last


@dataclass
class MultiscaleHierarchicalGatedMemoryResult:
    output: list
    fast_states: list
    scale_states: dict[str, list]
    spatial_attention_matrices: list[list[list[float]]]
    temporal_weights: list[list[float]]
    gates: dict[str, list]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def multiscale_hierarchical_hybrid_graph_temporal_gated_memory(
    sequence: list | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_embeddings: list | np.ndarray,
    num_heads: int = 4,
    temporal_window: int = 1,
    scale_decays: list[float] | tuple[float, ...] = (0.85, 0.95),
    gate_bias: float = 0.0,
    edge_embedding_scale: float = 1.0,
    temporal_decay: float = 1.0,
) -> MultiscaleHierarchicalGatedMemoryResult:
    """Hybrid attention with one fast state and multiple slow states."""
    base = hybrid_node_edge_temporal_attention(
        sequence,
        adjacency,
        edge_embeddings,
        num_heads=num_heads,
        temporal_window=temporal_window,
        activation="linear",
        edge_embedding_scale=edge_embedding_scale,
        temporal_decay=temporal_decay,
    )
    x = np.asarray(base.output, dtype=np.float64)
    h_fast = np.zeros_like(x[0])
    slow_states = [np.zeros_like(x[0]) for _ in scale_decays]
    outputs=[]
    fast_hist=[]
    slow_hist={f"scale_{i}": [] for i in range(len(scale_decays))}
    gates={f"scale_{i}": [] for i in range(len(scale_decays))}
    gates["fast"]=[]
    for t in range(x.shape[0]):
        xt=x[t]
        z_fast = _sigmoid(xt + h_fast + gate_bias)
        cand_fast = np.tanh(xt + h_fast)
        h_fast = (1.0 - z_fast) * h_fast + z_fast * cand_fast
        gates["fast"].append(z_fast.tolist())
        combined = h_fast.copy()
        for idx, decay in enumerate(scale_decays):
            z = _sigmoid(h_fast + decay * slow_states[idx] + gate_bias / (idx + 2))
            cand = np.tanh(h_fast + slow_states[idx])
            slow_states[idx] = (1.0 - z) * slow_states[idx] + z * cand
            slow_hist[f"scale_{idx}"].append(slow_states[idx].tolist())
            gates[f"scale_{idx}"].append(z.tolist())
            combined = combined + slow_states[idx]
        combined = combined / (1.0 + len(scale_decays))
        outputs.append(combined.tolist())
        fast_hist.append(h_fast.tolist())
    return MultiscaleHierarchicalGatedMemoryResult(
        output=outputs,
        fast_states=fast_hist,
        scale_states=slow_hist,
        spatial_attention_matrices=base.spatial_attention_matrices,
        temporal_weights=base.temporal_weights,
        gates=gates,
        meta={
            "time_steps": int(x.shape[0]),
            "nodes": int(x.shape[1]),
            "channels": int(x.shape[2]),
            "num_heads": num_heads,
            "temporal_window": temporal_window,
            "scale_decays": list(scale_decays),
            "gate_bias": gate_bias,
            "edge_embedding_scale": edge_embedding_scale,
            "temporal_decay": temporal_decay,
            "multiscale": True,
        },
    )



def multiscale_hierarchical_hybrid_graph_temporal_gated_stack(
    sequence: list | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_embeddings: list | np.ndarray,
    depth: int = 2,
    num_heads: int = 4,
    temporal_window: int = 1,
    scale_decays: list[float] | tuple[float, ...] = (0.85, 0.95),
    normalization: str = "layernorm",
    activation: str = "gelu",
    residual: bool = True,
    gate_bias: float = 0.0,
    edge_embedding_scale: float = 1.0,
    temporal_decay: float = 1.0,
) -> MultiscaleHierarchicalGatedMemoryResult:
    """Stack multiscale hierarchical gated memory blocks."""
    depth = ensure_positive_int(depth, "depth")
    current = np.asarray(sequence, dtype=np.float64)
    last: MultiscaleHierarchicalGatedMemoryResult | None = None
    for _ in range(depth):
        base = multiscale_hierarchical_hybrid_graph_temporal_gated_memory(
            current,
            adjacency,
            edge_embeddings,
            num_heads=num_heads,
            temporal_window=temporal_window,
            scale_decays=scale_decays,
            gate_bias=gate_bias,
            edge_embedding_scale=edge_embedding_scale,
            temporal_decay=temporal_decay,
        )
        y = np.asarray(base.output, dtype=np.float64)
        z=[]
        for t in range(y.shape[0]):
            block = np.asarray(graph_block_normalize_multichannel(y[t], mode=normalization), dtype=np.float64)
            ff = _apply_activation(block, activation)
            if residual and ff.shape == block.shape:
                ff = ff + block
            z.append(np.asarray(graph_block_normalize_multichannel(ff, mode=normalization), dtype=np.float64))
        current = np.stack(z, axis=0)
        last = MultiscaleHierarchicalGatedMemoryResult(
            output=current.tolist(),
            fast_states=base.fast_states,
            scale_states=base.scale_states,
            spatial_attention_matrices=base.spatial_attention_matrices,
            temporal_weights=base.temporal_weights,
            gates=base.gates,
            meta={
                "depth": depth,
                "num_heads": num_heads,
                "temporal_window": temporal_window,
                "scale_decays": list(scale_decays),
                "normalization": normalization,
                "activation": activation,
                "residual": residual,
                "gate_bias": gate_bias,
                "edge_embedding_scale": edge_embedding_scale,
                "temporal_decay": temporal_decay,
                "multiscale": True,
            },
        )
    assert last is not None
    return last


@dataclass
class AttentiveMultiscaleHierarchicalGatedMemoryResult:
    output: list
    fast_states: list
    scale_states: dict[str, list]
    scale_attention_weights: list[list[float]]
    spatial_attention_matrices: list[list[list[float]]]
    temporal_weights: list[list[float]]
    gates: dict[str, list]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def attentive_multiscale_hierarchical_hybrid_graph_temporal_gated_memory(
    sequence: list | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_embeddings: list | np.ndarray,
    num_heads: int = 4,
    temporal_window: int = 1,
    scale_decays: list[float] | tuple[float, ...] = (0.85, 0.95),
    gate_bias: float = 0.0,
    edge_embedding_scale: float = 1.0,
    temporal_decay: float = 1.0,
) -> AttentiveMultiscaleHierarchicalGatedMemoryResult:
    """Multiscale gated memory with soft attention across fast/slow states."""
    base = multiscale_hierarchical_hybrid_graph_temporal_gated_memory(
        sequence,
        adjacency,
        edge_embeddings,
        num_heads=num_heads,
        temporal_window=temporal_window,
        scale_decays=scale_decays,
        gate_bias=gate_bias,
        edge_embedding_scale=edge_embedding_scale,
        temporal_decay=temporal_decay,
    )
    fast = [np.asarray(x, dtype=np.float64) for x in base.fast_states]
    slow = {k: [np.asarray(v, dtype=np.float64) for v in vals] for k, vals in base.scale_states.items()}
    outputs=[]
    attn_hist=[]
    for t in range(len(fast)):
        reps=[fast[t]] + [slow[key][t] for key in sorted(slow)]
        scores=np.array([float(np.mean(np.abs(r))) for r in reps], dtype=np.float64)
        attn=np.exp(scores - np.max(scores))
        attn=attn / np.maximum(np.sum(attn), 1e-12)
        out=sum(w * r for w, r in zip(attn, reps, strict=False))
        outputs.append(out.tolist())
        attn_hist.append(attn.tolist())
    return AttentiveMultiscaleHierarchicalGatedMemoryResult(
        output=outputs,
        fast_states=base.fast_states,
        scale_states=base.scale_states,
        scale_attention_weights=attn_hist,
        spatial_attention_matrices=base.spatial_attention_matrices,
        temporal_weights=base.temporal_weights,
        gates=base.gates,
        meta={**base.meta, "attentive_multiscale": True, "num_scales": 1 + len(scale_decays)},
    )



def attentive_multiscale_hierarchical_hybrid_graph_temporal_gated_stack(
    sequence: list | np.ndarray,
    adjacency: list[list[float]] | np.ndarray,
    edge_embeddings: list | np.ndarray,
    depth: int = 2,
    num_heads: int = 4,
    temporal_window: int = 1,
    scale_decays: list[float] | tuple[float, ...] = (0.85, 0.95),
    normalization: str = "layernorm",
    activation: str = "gelu",
    residual: bool = True,
    gate_bias: float = 0.0,
    edge_embedding_scale: float = 1.0,
    temporal_decay: float = 1.0,
) -> AttentiveMultiscaleHierarchicalGatedMemoryResult:
    """Stack attentive multiscale gated memory blocks."""
    depth = ensure_positive_int(depth, "depth")
    current = np.asarray(sequence, dtype=np.float64)
    last: AttentiveMultiscaleHierarchicalGatedMemoryResult | None = None
    for _ in range(depth):
        base = attentive_multiscale_hierarchical_hybrid_graph_temporal_gated_memory(
            current,
            adjacency,
            edge_embeddings,
            num_heads=num_heads,
            temporal_window=temporal_window,
            scale_decays=scale_decays,
            gate_bias=gate_bias,
            edge_embedding_scale=edge_embedding_scale,
            temporal_decay=temporal_decay,
        )
        y=np.asarray(base.output, dtype=np.float64)
        z=[]
        for t in range(y.shape[0]):
            block=np.asarray(graph_block_normalize_multichannel(y[t], mode=normalization), dtype=np.float64)
            ff=_apply_activation(block, activation)
            if residual and ff.shape == block.shape:
                ff = ff + block
            z.append(np.asarray(graph_block_normalize_multichannel(ff, mode=normalization), dtype=np.float64))
        current=np.stack(z, axis=0)
        last = AttentiveMultiscaleHierarchicalGatedMemoryResult(
            output=current.tolist(),
            fast_states=base.fast_states,
            scale_states=base.scale_states,
            scale_attention_weights=base.scale_attention_weights,
            spatial_attention_matrices=base.spatial_attention_matrices,
            temporal_weights=base.temporal_weights,
            gates=base.gates,
            meta={**base.meta, "depth": depth, "normalization": normalization, "activation": activation, "residual": residual, "attentive_multiscale": True},
        )
    assert last is not None
    return last


__all__ = [
    "GNNStackResult",
    "MultiHeadAttentionResult",
    "MultiHeadNodeAttentionResult",
    "DeepGNNResult",
    "GraphTransformerResult",
    "EdgeConditionedConvResult",
    "MessagePassingResult",
    "QKVAttentionResult",
    "ChannelMixResult",
    "HybridTemporalAttentionResult",
    "HybridGatedMemoryResult",
    "BidirectionalGatedMemoryResult",
    "HierarchicalGatedMemoryResult",
    "MultiscaleHierarchicalGatedMemoryResult",
    "AttentiveMultiscaleHierarchicalGatedMemoryResult",
    "RecurrentHybridAttentionResult",
    "graph_block_normalize",
    "channel_mix",
    "graph_block_normalize_multichannel",
    "edge_aware_message_passing",
    "edge_conditioned_convolution",
    "edge_conditioned_conv_stack",
    "edge_feature_message_passing_stack",
    "graph_pool",
    "graph_attention_matrix",
    "graph_attention_filter",
    "multihead_graph_attention",
    "multihead_graph_attention_multichannel",
    "qkv_graph_attention",
    "masked_qkv_graph_attention",
    "spectral_gnn_filter",
    "spectral_gnn_filter_multichannel",
    "graph_scattering_transform",
    "stacked_gnn",
    "deep_gnn_stack",
    "graph_transformer_layer",
    "graph_transformer_stack",
    "graph_transformer_qkv_layer",
    "graph_transformer_qkv_stack",
    "graph_transformer_masked_qkv_layer",
    "graph_transformer_masked_qkv_stack",
    "graph_transformer_enhanced_layer",
    "graph_transformer_enhanced_stack",
    "structured_edge_embedding_attention",
    "graph_transformer_edge_embedding_layer",
    "graph_transformer_edge_embedding_stack",
    "hybrid_node_edge_temporal_attention",
    "hybrid_graph_temporal_transformer_layer",
    "hybrid_graph_temporal_transformer_stack",
    "hybrid_graph_temporal_gated_stack",
    "bidirectional_hybrid_graph_temporal_gated_stack",
    "hierarchical_hybrid_graph_temporal_gated_stack",
    "multiscale_hierarchical_hybrid_graph_temporal_gated_stack",
    "attentive_multiscale_hierarchical_hybrid_graph_temporal_gated_stack",
    "attentive_multiscale_hierarchical_hybrid_graph_temporal_gated_memory",
    "multiscale_hierarchical_hybrid_graph_temporal_gated_memory",
    "hierarchical_hybrid_graph_temporal_gated_memory",
    "bidirectional_hybrid_graph_temporal_gated_memory",
    "hybrid_graph_temporal_gated_memory",
    "recurrent_hybrid_node_edge_temporal_attention",
    "recurrent_hybrid_graph_temporal_transformer_stack",
]
