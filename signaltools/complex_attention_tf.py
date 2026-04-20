"""Complex multiband attention with coupling between heads in the time-frequency domain."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .complex_multihead_tf import multihead_band_complex_tf_operator
from .utils import ensure_positive_int


@dataclass
class ComplexCoupledAttentionTFResult:
    time_real: list[list[float]]
    time_imag: list[list[float]]
    coupled_specs_real: list
    coupled_specs_imag: list
    attention_weights: list[list[float]]
    band_assignments: list[int]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _window(frame_size: int, window: str) -> np.ndarray:
    if window == "hann":
        return np.hanning(frame_size).astype(np.float64)
    if window == "rect":
        return np.ones(frame_size, dtype=np.float64)
    raise ValueError("Unsupported window")



def _reconstruct_from_specs(specs: np.ndarray, samples: int, hop_size: int, window: str) -> np.ndarray:
    frame_size = specs.shape[1]
    w = _window(frame_size, window)
    recon = np.zeros((samples, specs.shape[2]), dtype=np.complex128)
    norm = np.zeros(samples, dtype=np.float64)
    for frame_idx in range(specs.shape[0]):
        start = frame_idx * hop_size
        end = min(start + frame_size, samples)
        time_frame = np.fft.ifft(specs[frame_idx], axis=0)
        valid = end - start
        recon[start:end] += time_frame[:valid]
        norm[start:end] += w[:valid] ** 2
    return recon / np.maximum(norm[:, None], 1e-12)



def complex_multiband_head_coupling_operator(
    features: list | np.ndarray,
    frame_size: int = 256,
    hop_size: int = 128,
    num_heads: int = 4,
    head_gains: list | np.ndarray | None = None,
    head_biases: list | np.ndarray | None = None,
    phase_shifts: list[float] | np.ndarray | None = None,
    coupling_matrix: list[list[float]] | np.ndarray | None = None,
    residual: bool = True,
    window: str = "hann",
) -> ComplexCoupledAttentionTFResult:
    """Apply multi-head banded TF operators, then couple heads with a learnable-style matrix."""
    num_heads = ensure_positive_int(num_heads, "num_heads")
    base = multihead_band_complex_tf_operator(
        features,
        frame_size=frame_size,
        hop_size=hop_size,
        num_heads=num_heads,
        head_gains=head_gains,
        head_biases=head_biases,
        phase_shifts=phase_shifts,
        residual=residual,
        combine="mean",
        window=window,
    )
    head_specs = np.asarray(base.head_outputs_real, dtype=np.float64) + 1j * np.asarray(base.head_outputs_imag, dtype=np.float64)
    if coupling_matrix is None:
        coupling = 0.85 * np.eye(num_heads, dtype=np.float64) + 0.15 / max(num_heads, 1) * np.ones((num_heads, num_heads), dtype=np.float64)
    else:
        coupling = np.asarray(coupling_matrix, dtype=np.float64)
        if coupling.shape != (num_heads, num_heads):
            raise ValueError("coupling_matrix must have shape (num_heads, num_heads)")
    attn = np.exp(coupling - np.max(coupling, axis=1, keepdims=True))
    attn = attn / np.maximum(np.sum(attn, axis=1, keepdims=True), 1e-12)
    coupled = np.einsum("ab,bftc->aftc", attn, head_specs)
    combined_specs = np.mean(coupled, axis=0)
    samples = len(base.time_real)
    recon = _reconstruct_from_specs(combined_specs, samples=samples, hop_size=hop_size, window=window)
    return ComplexCoupledAttentionTFResult(
        time_real=np.real(recon).tolist(),
        time_imag=np.imag(recon).tolist(),
        coupled_specs_real=np.real(coupled).tolist(),
        coupled_specs_imag=np.imag(coupled).tolist(),
        attention_weights=attn.tolist(),
        band_assignments=base.band_assignments,
        meta={
            "samples": samples,
            "channels": int(recon.shape[1]),
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "num_heads": int(num_heads),
            "residual": residual,
            "window": window,
        },
    )



def complex_multiband_head_coupling_stack(
    features: list | np.ndarray,
    depth: int = 2,
    **kwargs: Any,
) -> ComplexCoupledAttentionTFResult:
    """Apply several coupled-head TF layers in sequence."""
    depth = ensure_positive_int(depth, "depth")
    current = np.asarray(features, dtype=np.complex128)
    if current.ndim == 1:
        current = current[:, None]
    last: ComplexCoupledAttentionTFResult | None = None
    for _ in range(depth):
        last = complex_multiband_head_coupling_operator(current, **kwargs)
        current = np.asarray(last.time_real, dtype=np.float64) + 1j * np.asarray(last.time_imag, dtype=np.float64)
    assert last is not None
    last.meta["depth"] = depth
    return last



def temporal_complex_head_coupling_operator(
    features: list | np.ndarray,
    frame_size: int = 256,
    hop_size: int = 128,
    num_heads: int = 4,
    head_gains: list | np.ndarray | None = None,
    head_biases: list | np.ndarray | None = None,
    phase_shifts: list[float] | np.ndarray | None = None,
    coupling_schedule: list | np.ndarray | None = None,
    residual: bool = True,
    window: str = "hann",
) -> ComplexCoupledAttentionTFResult:
    """Apply frame-varying coupling matrices between complex TF heads."""
    base = multihead_band_complex_tf_operator(
        features,
        frame_size=frame_size,
        hop_size=hop_size,
        num_heads=num_heads,
        head_gains=head_gains,
        head_biases=head_biases,
        phase_shifts=phase_shifts,
        residual=residual,
        combine="mean",
        window=window,
    )
    head_specs = np.asarray(base.head_outputs_real, dtype=np.float64) + 1j * np.asarray(base.head_outputs_imag, dtype=np.float64)
    heads, frames = head_specs.shape[0], head_specs.shape[1]
    if coupling_schedule is None:
        schedule = np.stack([
            0.85 * np.eye(heads, dtype=np.float64) + (0.15 + 0.05 * np.sin(2 * np.pi * f / max(frames, 1))) / max(heads, 1) * np.ones((heads, heads), dtype=np.float64)
            for f in range(frames)
        ], axis=0)
    else:
        schedule = np.asarray(coupling_schedule, dtype=np.float64)
        if schedule.shape != (frames, heads, heads):
            raise ValueError("coupling_schedule must have shape (frames, num_heads, num_heads)")
    coupled = np.zeros_like(head_specs)
    weights = np.zeros_like(schedule)
    for frame_idx in range(frames):
        attn = np.exp(schedule[frame_idx] - np.max(schedule[frame_idx], axis=1, keepdims=True))
        attn = attn / np.maximum(np.sum(attn, axis=1, keepdims=True), 1e-12)
        weights[frame_idx] = attn
        coupled[:, frame_idx] = np.einsum('ab,bfc->afc', attn, head_specs[:, frame_idx])
    combined_specs = np.mean(coupled, axis=0)
    samples = len(base.time_real)
    recon = _reconstruct_from_specs(combined_specs, samples=samples, hop_size=hop_size, window=window)
    return ComplexCoupledAttentionTFResult(
        time_real=np.real(recon).tolist(),
        time_imag=np.imag(recon).tolist(),
        coupled_specs_real=np.real(coupled).tolist(),
        coupled_specs_imag=np.imag(coupled).tolist(),
        attention_weights=weights.tolist(),
        band_assignments=base.band_assignments,
        meta={
            "samples": samples,
            "channels": int(recon.shape[1]),
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "num_heads": int(num_heads),
            "frames": int(frames),
            "time_dependent": True,
            "residual": residual,
            "window": window,
        },
    )



def content_conditioned_temporal_head_coupling_operator(
    features: list | np.ndarray,
    frame_size: int = 256,
    hop_size: int = 128,
    num_heads: int = 4,
    head_gains: list | np.ndarray | None = None,
    head_biases: list | np.ndarray | None = None,
    phase_shifts: list[float] | np.ndarray | None = None,
    residual: bool = True,
    window: str = "hann",
) -> ComplexCoupledAttentionTFResult:
    """Build frame-wise head couplings from the content of each frame."""
    base = multihead_band_complex_tf_operator(
        features,
        frame_size=frame_size,
        hop_size=hop_size,
        num_heads=num_heads,
        head_gains=head_gains,
        head_biases=head_biases,
        phase_shifts=phase_shifts,
        residual=residual,
        combine="mean",
        window=window,
    )
    head_specs = np.asarray(base.head_outputs_real, dtype=np.float64) + 1j * np.asarray(base.head_outputs_imag, dtype=np.float64)
    heads, frames = head_specs.shape[0], head_specs.shape[1]
    coupled = np.zeros_like(head_specs)
    weights = np.zeros((frames, heads, heads), dtype=np.float64)
    for frame_idx in range(frames):
        frame_repr = np.abs(head_specs[:, frame_idx]).reshape(heads, -1)
        norms = np.linalg.norm(frame_repr, axis=1, keepdims=True)
        sims = frame_repr @ frame_repr.T / np.maximum(norms @ norms.T, 1e-12)
        sims = sims + 0.1 * np.eye(heads, dtype=np.float64)
        attn = np.exp(sims - np.max(sims, axis=1, keepdims=True))
        attn = attn / np.maximum(np.sum(attn, axis=1, keepdims=True), 1e-12)
        weights[frame_idx] = attn
        coupled[:, frame_idx] = np.einsum('ab,bfc->afc', attn, head_specs[:, frame_idx])
    combined_specs = np.mean(coupled, axis=0)
    samples = len(base.time_real)
    recon = _reconstruct_from_specs(combined_specs, samples=samples, hop_size=hop_size, window=window)
    return ComplexCoupledAttentionTFResult(
        time_real=np.real(recon).tolist(),
        time_imag=np.imag(recon).tolist(),
        coupled_specs_real=np.real(coupled).tolist(),
        coupled_specs_imag=np.imag(coupled).tolist(),
        attention_weights=weights.tolist(),
        band_assignments=base.band_assignments,
        meta={
            "samples": samples,
            "channels": int(recon.shape[1]),
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "num_heads": int(num_heads),
            "frames": int(frames),
            "content_conditioned": True,
            "residual": residual,
            "window": window,
        },
    )



def mode_conditioned_temporal_head_coupling_operator(
    features: list | np.ndarray,
    frame_size: int = 256,
    hop_size: int = 128,
    num_heads: int = 4,
    head_gains: list | np.ndarray | None = None,
    head_biases: list | np.ndarray | None = None,
    phase_shifts: list[float] | np.ndarray | None = None,
    mode: str = "causal",
    residual: bool = True,
    window: str = "hann",
) -> ComplexCoupledAttentionTFResult:
    """Frame-varying coupling using causal or noncausal temporal context."""
    base = multihead_band_complex_tf_operator(
        features,
        frame_size=frame_size,
        hop_size=hop_size,
        num_heads=num_heads,
        head_gains=head_gains,
        head_biases=head_biases,
        phase_shifts=phase_shifts,
        residual=residual,
        combine="mean",
        window=window,
    )
    head_specs = np.asarray(base.head_outputs_real, dtype=np.float64) + 1j * np.asarray(base.head_outputs_imag, dtype=np.float64)
    heads, frames = head_specs.shape[0], head_specs.shape[1]
    coupled = np.zeros_like(head_specs)
    weights = np.zeros((frames, heads, heads), dtype=np.float64)
    for frame_idx in range(frames):
        if mode == "causal":
            ctx = head_specs[:, : frame_idx + 1]
        elif mode == "noncausal":
            start = max(0, frame_idx - 1)
            stop = min(frames, frame_idx + 2)
            ctx = head_specs[:, start:stop]
        else:
            raise ValueError("mode must be 'causal' or 'noncausal'")
        frame_repr = np.abs(ctx).reshape(heads, -1)
        norms = np.linalg.norm(frame_repr, axis=1, keepdims=True)
        sims = frame_repr @ frame_repr.T / np.maximum(norms @ norms.T, 1e-12)
        sims = sims + 0.1 * np.eye(heads, dtype=np.float64)
        attn = np.exp(sims - np.max(sims, axis=1, keepdims=True))
        attn = attn / np.maximum(np.sum(attn, axis=1, keepdims=True), 1e-12)
        weights[frame_idx] = attn
        coupled[:, frame_idx] = np.einsum('ab,bfc->afc', attn, head_specs[:, frame_idx])
    combined_specs = np.mean(coupled, axis=0)
    samples = len(base.time_real)
    recon = _reconstruct_from_specs(combined_specs, samples=samples, hop_size=hop_size, window=window)
    return ComplexCoupledAttentionTFResult(
        time_real=np.real(recon).tolist(),
        time_imag=np.imag(recon).tolist(),
        coupled_specs_real=np.real(coupled).tolist(),
        coupled_specs_imag=np.imag(coupled).tolist(),
        attention_weights=weights.tolist(),
        band_assignments=base.band_assignments,
        meta={
            "samples": samples,
            "channels": int(recon.shape[1]),
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "num_heads": int(num_heads),
            "frames": int(frames),
            "temporal_mode": mode,
            "residual": residual,
            "window": window,
        },
    )



def long_memory_temporal_head_coupling_operator(
    features: list | np.ndarray,
    frame_size: int = 256,
    hop_size: int = 128,
    num_heads: int = 4,
    head_gains: list | np.ndarray | None = None,
    head_biases: list | np.ndarray | None = None,
    phase_shifts: list[float] | np.ndarray | None = None,
    memory_decay: float = 0.9,
    residual: bool = True,
    window: str = "hann",
) -> ComplexCoupledAttentionTFResult:
    """Temporal head coupling with exponentially decayed long-range memory."""
    base = multihead_band_complex_tf_operator(
        features,
        frame_size=frame_size,
        hop_size=hop_size,
        num_heads=num_heads,
        head_gains=head_gains,
        head_biases=head_biases,
        phase_shifts=phase_shifts,
        residual=residual,
        combine="mean",
        window=window,
    )
    head_specs = np.asarray(base.head_outputs_real, dtype=np.float64) + 1j * np.asarray(base.head_outputs_imag, dtype=np.float64)
    heads, frames = head_specs.shape[0], head_specs.shape[1]
    coupled = np.zeros_like(head_specs)
    weights = np.zeros((frames, heads, heads), dtype=np.float64)
    memory = np.abs(head_specs[:, 0]).reshape(heads, -1)
    for frame_idx in range(frames):
        current = np.abs(head_specs[:, frame_idx]).reshape(heads, -1)
        memory = memory_decay * memory + (1.0 - memory_decay) * current
        norms = np.linalg.norm(memory, axis=1, keepdims=True)
        sims = memory @ memory.T / np.maximum(norms @ norms.T, 1e-12)
        sims = sims + 0.1 * np.eye(heads, dtype=np.float64)
        attn = np.exp(sims - np.max(sims, axis=1, keepdims=True))
        attn = attn / np.maximum(np.sum(attn, axis=1, keepdims=True), 1e-12)
        weights[frame_idx] = attn
        coupled[:, frame_idx] = np.einsum('ab,bfc->afc', attn, head_specs[:, frame_idx])
    combined_specs = np.mean(coupled, axis=0)
    samples = len(base.time_real)
    recon = _reconstruct_from_specs(combined_specs, samples=samples, hop_size=hop_size, window=window)
    return ComplexCoupledAttentionTFResult(
        time_real=np.real(recon).tolist(),
        time_imag=np.imag(recon).tolist(),
        coupled_specs_real=np.real(coupled).tolist(),
        coupled_specs_imag=np.imag(coupled).tolist(),
        attention_weights=weights.tolist(),
        band_assignments=base.band_assignments,
        meta={
            "samples": samples,
            "channels": int(recon.shape[1]),
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "num_heads": int(num_heads),
            "frames": int(frames),
            "memory_decay": float(memory_decay),
            "long_memory": True,
            "residual": residual,
            "window": window,
        },
    )



def stability_regularized_temporal_head_coupling_operator(
    features: list | np.ndarray,
    frame_size: int = 256,
    hop_size: int = 128,
    num_heads: int = 4,
    head_gains: list | np.ndarray | None = None,
    head_biases: list | np.ndarray | None = None,
    phase_shifts: list[float] | np.ndarray | None = None,
    stability_lambda: float = 0.5,
    residual: bool = True,
    window: str = "hann",
) -> ComplexCoupledAttentionTFResult:
    """Temporal coupling with an explicit penalty on frame-to-frame attention changes."""
    base = multihead_band_complex_tf_operator(
        features,
        frame_size=frame_size,
        hop_size=hop_size,
        num_heads=num_heads,
        head_gains=head_gains,
        head_biases=head_biases,
        phase_shifts=phase_shifts,
        residual=residual,
        combine="mean",
        window=window,
    )
    head_specs = np.asarray(base.head_outputs_real, dtype=np.float64) + 1j * np.asarray(base.head_outputs_imag, dtype=np.float64)
    heads, frames = head_specs.shape[0], head_specs.shape[1]
    coupled = np.zeros_like(head_specs)
    weights = np.zeros((frames, heads, heads), dtype=np.float64)
    prev_attn = np.eye(heads, dtype=np.float64)
    for frame_idx in range(frames):
        frame_repr = np.abs(head_specs[:, frame_idx]).reshape(heads, -1)
        norms = np.linalg.norm(frame_repr, axis=1, keepdims=True)
        sims = frame_repr @ frame_repr.T / np.maximum(norms @ norms.T, 1e-12)
        raw = np.exp(sims - np.max(sims, axis=1, keepdims=True))
        raw = raw / np.maximum(np.sum(raw, axis=1, keepdims=True), 1e-12)
        attn = (1.0 - stability_lambda) * raw + stability_lambda * prev_attn
        attn = attn / np.maximum(np.sum(attn, axis=1, keepdims=True), 1e-12)
        weights[frame_idx] = attn
        prev_attn = attn
        coupled[:, frame_idx] = np.einsum('ab,bfc->afc', attn, head_specs[:, frame_idx])
    combined_specs = np.mean(coupled, axis=0)
    samples = len(base.time_real)
    recon = _reconstruct_from_specs(combined_specs, samples=samples, hop_size=hop_size, window=window)
    return ComplexCoupledAttentionTFResult(
        time_real=np.real(recon).tolist(),
        time_imag=np.imag(recon).tolist(),
        coupled_specs_real=np.real(coupled).tolist(),
        coupled_specs_imag=np.imag(coupled).tolist(),
        attention_weights=weights.tolist(),
        band_assignments=base.band_assignments,
        meta={
            "samples": samples,
            "channels": int(recon.shape[1]),
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "num_heads": int(num_heads),
            "frames": int(frames),
            "stability_lambda": float(stability_lambda),
            "temporal_stability_regularized": True,
            "residual": residual,
            "window": window,
        },
    )



def joint_temporal_spectral_regularized_coupling_operator(
    features: list | np.ndarray,
    frame_size: int = 256,
    hop_size: int = 128,
    num_heads: int = 4,
    head_gains: list | np.ndarray | None = None,
    head_biases: list | np.ndarray | None = None,
    phase_shifts: list[float] | np.ndarray | None = None,
    temporal_lambda: float = 0.5,
    spectral_lambda: float = 0.1,
    residual: bool = True,
    window: str = "hann",
) -> ComplexCoupledAttentionTFResult:
    """Coupling with temporal attention smoothing and spectral-bin smoothing."""
    base = multihead_band_complex_tf_operator(
        features,
        frame_size=frame_size,
        hop_size=hop_size,
        num_heads=num_heads,
        head_gains=head_gains,
        head_biases=head_biases,
        phase_shifts=phase_shifts,
        residual=residual,
        combine="mean",
        window=window,
    )
    head_specs = np.asarray(base.head_outputs_real, dtype=np.float64) + 1j * np.asarray(base.head_outputs_imag, dtype=np.float64)
    heads, frames = head_specs.shape[0], head_specs.shape[1]
    coupled = np.zeros_like(head_specs)
    weights = np.zeros((frames, heads, heads), dtype=np.float64)
    prev_attn = np.eye(heads, dtype=np.float64)
    for frame_idx in range(frames):
        frame_repr = np.abs(head_specs[:, frame_idx]).reshape(heads, -1)
        norms = np.linalg.norm(frame_repr, axis=1, keepdims=True)
        sims = frame_repr @ frame_repr.T / np.maximum(norms @ norms.T, 1e-12)
        raw = np.exp(sims - np.max(sims, axis=1, keepdims=True))
        raw = raw / np.maximum(np.sum(raw, axis=1, keepdims=True), 1e-12)
        attn = (1.0 - temporal_lambda) * raw + temporal_lambda * prev_attn
        attn = attn / np.maximum(np.sum(attn, axis=1, keepdims=True), 1e-12)
        weights[frame_idx] = attn
        prev_attn = attn
        coupled[:, frame_idx] = np.einsum('ab,bfc->afc', attn, head_specs[:, frame_idx])
    combined_specs = np.mean(coupled, axis=0)
    if combined_specs.shape[1] > 1 and spectral_lambda > 0.0:
        left = np.roll(combined_specs, 1, axis=1)
        right = np.roll(combined_specs, -1, axis=1)
        combined_specs = (1.0 - spectral_lambda) * combined_specs + 0.5 * spectral_lambda * (left + right)
    samples = len(base.time_real)
    recon = _reconstruct_from_specs(combined_specs, samples=samples, hop_size=hop_size, window=window)
    return ComplexCoupledAttentionTFResult(
        time_real=np.real(recon).tolist(),
        time_imag=np.imag(recon).tolist(),
        coupled_specs_real=np.real(coupled).tolist(),
        coupled_specs_imag=np.imag(coupled).tolist(),
        attention_weights=weights.tolist(),
        band_assignments=base.band_assignments,
        meta={
            "samples": samples,
            "channels": int(recon.shape[1]),
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "num_heads": int(num_heads),
            "frames": int(frames),
            "temporal_lambda": float(temporal_lambda),
            "spectral_lambda": float(spectral_lambda),
            "joint_temporal_spectral_regularized": True,
            "residual": residual,
            "window": window,
        },
    )


__all__ = [
    "joint_temporal_spectral_regularized_coupling_operator",
    "stability_regularized_temporal_head_coupling_operator",
    "long_memory_temporal_head_coupling_operator",
    "mode_conditioned_temporal_head_coupling_operator",
    "content_conditioned_temporal_head_coupling_operator",
    "temporal_complex_head_coupling_operator",
    "ComplexCoupledAttentionTFResult",
    "complex_multiband_head_coupling_operator",
    "complex_multiband_head_coupling_stack",
]
