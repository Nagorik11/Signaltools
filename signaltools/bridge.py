from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import struct
import math
import random
from .framing import FrameConfig, frame_signal, normalize_signal, detrend_mean
from .features import frame_feature_vector, first_derivative, second_derivative
from .spectral import dft, dominant_bins, spectral_energy, spectral_flatness
from .detect import threshold_events, local_peaks
from .bitlayer import build_bit_signature, compact_bit_expression, bytes_to_bits

@dataclass
class LayeredSignalAnalysis:
    physical_layer: dict[str, Any]
    digital_layer: dict[str, Any]
    temporal_layer: dict[str, Any]
    spectral_layer: dict[str, Any]
    symbolic_layer: dict[str, Any]
    latent_layer: dict[str, Any]
    classification: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class SignalSignature:
    samples: int
    frames: int
    mean_frame_rms: float
    mean_frame_zcr: float
    mean_frame_variance: float
    dominant_bins: list[dict[str, float]]
    spectral_energy: float
    spectral_flatness: float
    event_count: int
    peak_count: int
    derivative_energy: float
    curvature_energy: float
    meta: dict[str, Any]
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def signal_signature(signal: list[float] | list[int], frame_size: int = 256, hop_size: int = 128, threshold: float = 0.35) -> SignalSignature:
    s = detrend_mean(normalize_signal(signal))
    frames = frame_signal(s, FrameConfig(frame_size=frame_size, hop_size=hop_size))
    feats = [frame_feature_vector(f) for f in frames] if frames else []
    rms_vals = [f["rms"] for f in feats]
    zcr_vals = [f["zcr"] for f in feats]
    var_vals = [f["variance"] for f in feats]
    ref = frames[0] if frames else s[:frame_size]
    spec = dft(ref) if ref else {"magnitude": []}
    dom = dominant_bins(spec["magnitude"], top_k=8) if ref else []
    events = threshold_events(s, threshold)
    peaks = local_peaks(s, threshold)
    d1, d2 = first_derivative(s), second_derivative(s)
    return SignalSignature(
        samples=len(s), frames=len(frames),
        mean_frame_rms=round(_mean(rms_vals), 6),
        mean_frame_zcr=round(_mean(zcr_vals), 6),
        mean_frame_variance=round(_mean(var_vals), 6),
        dominant_bins=[{"bin": int(x["bin"]), "magnitude": float(x["magnitude"])} for x in dom],
        spectral_energy=spectral_energy(spec["magnitude"]) if ref else 0.0,
        spectral_flatness=spectral_flatness(spec["magnitude"]) if ref else 0.0,
        event_count=len(events), peak_count=len(peaks),
        derivative_energy=round(sum(x*x for x in d1), 6),
        curvature_energy=round(sum(x*x for x in d2), 6),
        meta={"frame_size": frame_size, "hop_size": hop_size, "threshold": threshold},
    )

def signature_to_glyph_vector(sig: SignalSignature) -> list[float]:
    bins = sig.dominant_bins[:4]
    bin_positions = [float(b["bin"]) for b in bins]
    bin_mags = [float(b["magnitude"]) for b in bins]
    while len(bin_positions) < 4: bin_positions.append(0.0)
    while len(bin_mags) < 4: bin_mags.append(0.0)
    return [
        sig.mean_frame_rms, sig.mean_frame_zcr, sig.mean_frame_variance,
        sig.spectral_energy, sig.spectral_flatness,
        float(sig.event_count), float(sig.peak_count),
        sig.derivative_energy, sig.curvature_energy,
        *bin_positions, *bin_mags,
    ]

def analyze_signal_layered(signal: list[float] | list[int] | bytes, source_type: str = "unknown") -> LayeredSignalAnalysis:
    # 1. Handle signal input
    if isinstance(signal, bytes):
        raw_bytes = signal
        bits = bytes_to_bits(raw_bytes)
        # For bit-derived signals, we use +/- 1 mapping
        numeric_signal = [float(b) if b == 1 else -1.0 for b in bits]
    else:
        numeric_signal = signal
        # For numeric signals, we pack to bytes for bit analysis signature
        try:
            raw_bytes = struct.pack(f"<{len(numeric_signal)}f", *[float(x) for x in numeric_signal])
        except Exception:
            raw_bytes = bytes(len(numeric_signal))
        bits = bytes_to_bits(raw_bytes)

    # 2. Extract base components
    sig = signal_signature(numeric_signal)
    bit_sig = build_bit_signature(bits)
    
    # 3. Layered construction
    physical = {"source_type": source_type}
    
    digital = {
        "sample_count": len(numeric_signal),
        "data_type": "float32" if not isinstance(signal, bytes) else "raw_bytes",
        "bit_entropy": bit_sig.entropy
    }
    
    temporal = {
        "peak_count": sig.peak_count,
        "event_count": sig.event_count,
        "periodicity": "quasi-periodic" if sig.event_count > 2 else "stochastic"
    }
    
    spectral = {
        "dominant_bins": [b["bin"] for b in sig.dominant_bins[:4]],
        "harmonic_structure": "low-frequency dominant" if any(b["bin"] < 10 for b in sig.dominant_bins[:2]) else "broadband"
    }
    
    symbolic = {
        "compact_expression": compact_bit_expression(bit_sig)
    }
    
    latent = {
        "glyph_vector": signature_to_glyph_vector(sig)
    }
    
    # Simple classification heuristic
    family = "stochastic"
    if temporal["periodicity"] == "quasi-periodic":
        family = "periodic_structured"
    if sig.mean_frame_rms > 0.5:
        family = "high_energy_" + family
        
    classification = {"signal_family": family}
    
    return LayeredSignalAnalysis(
        physical_layer=physical,
        digital_layer=digital,
        temporal_layer=temporal,
        spectral_layer=spectral,
        symbolic_layer=symbolic,
        latent_layer=latent,
        classification=classification
    )

def reconstruct_signal_from_signature(sig: SignalSignature, duration: float = 1.0, sample_rate: int = 44100) -> list[float]:
    n = int(duration * sample_rate)
    if n <= 0: return []
    signal = [0.0] * n
    
    # Use dominant bins to synthesize sine waves
    frame_size = sig.meta.get("frame_size", 256)
    for binfo in sig.dominant_bins:
        freq = binfo["bin"] * sample_rate / frame_size
        mag = binfo["magnitude"] / (frame_size / 2.0)
        for t in range(n):
            signal[t] += mag * math.sin(2 * math.pi * freq * t / sample_rate)
            
    # Add noise based on variance
    std_dev = (sig.mean_frame_variance ** 0.5) if sig.mean_frame_variance > 0 else 0.02
    for t in range(n):
        signal[t] += (random.random() * 2 - 1) * std_dev * 0.3
        
    # Final RMS scaling to match original signal
    current_rms = (sum(x*x for x in signal) / n) ** 0.5
    if current_rms > 0:
        factor = sig.mean_frame_rms / current_rms
        signal = [x * factor for x in signal]
        
    return signal
