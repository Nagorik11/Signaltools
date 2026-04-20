from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import math

# Convert bytes to bits
def bytes_to_bits(raw: bytes) -> list[int]:
    bits = []
    for byte in raw:
        for i in range(8):
            bits.append((byte >> (7-i)) & 1)
    return bits
# Convert bits to signal
def bits_to_signal(bits: list[int]) -> list[float]:
    return [1.0 if b == 1 else -1.0 for b in bits]
# Calculate bit entropy
def bit_entropy(bits: list[int]) -> float:
    if not bits:
        return 0.0
    p1 = sum(bits) / len(bits)
    p0 = 1 - p1
    def h(p): return -p * math.log2(p) if p > 0 else 0.0
    return h(p0) + h(p1)
# Calculate bit transitions
def bit_transitions(bits: list[int]) -> int:
    return sum(1 for i in range(len(bits)-1) if bits[i] != bits[i+1])
# Calculate run lengths
def run_lengths(bits: list[int]) -> list[tuple[int, int]]:
    if not bits:
        return []
    runs, current, count = [], bits[0], 1
    for b in bits[1:]:
        if b == current:
            count += 1
        else:
            runs.append((current, count))
            current, count = b, 1
    runs.append((current, count))
    return runs
# Detect dominant periods
def detect_period(bits: list[int], max_period: int = 64) -> list[tuple[int, float]]:
    scores = {}
    for p in range(1, max_period):
        matches = sum(1 for i in range(len(bits)-p) if bits[i] == bits[i+p])
        if len(bits) - p > 0:
            scores[p] = matches / (len(bits)-p)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
# Calculate bit balance
def bit_balance(bits: list[int]) -> float:
    if not bits:
        return 0.0
    ones = sum(bits); zeros = len(bits)-ones
    return abs(ones-zeros) / len(bits)
# Calculate average run length
def average_run_length(runs: list[tuple[int, int]]) -> float:
    return sum(length for _, length in runs) / len(runs) if runs else 0.0
# Calculate longest run
def longest_run(runs: list[tuple[int, int]]) -> int:
    return max((length for _, length in runs), default=0)

@dataclass
class BitSignature:
    total_bits: int
    entropy: float
    transitions: int
    transition_density: float
    avg_run_length: float
    longest_run: int
    balance: float
    dominant_periods: list[tuple[int, float]]
    meta: dict[str, Any]
    
    def to_dict(self): return asdict(self)

def build_bit_signature(bits: list[int]) -> BitSignature:
    runs = run_lengths(bits)
    total = len(bits)
    transitions = bit_transitions(bits)
    # Optimization: periodic detection is expensive on large streams
    period_sample = bits[:10000] if total > 10000 else bits
    return BitSignature(
        total_bits=total,
        entropy=round(bit_entropy(bits), 6),
        transitions=transitions,
        transition_density=round(transitions / max(1, total), 6),
        avg_run_length=round(average_run_length(runs), 6),
        longest_run=longest_run(runs),
        balance=round(bit_balance(bits), 6),
        dominant_periods=detect_period(period_sample),
        meta={"runs_sample": runs[:10]},
    )

def compact_bit_expression(sig: BitSignature) -> str:
    periods = ",".join(f"{p}:{s:.2f}" for p, s in sig.dominant_periods[:3])
    return f"BIT{{N:{sig.total_bits}|H:{sig.entropy:.3f}|T:{sig.transitions}|D:{sig.transition_density:.3f}|R:{sig.avg_run_length:.2f}|L:{sig.longest_run}|B:{sig.balance:.2f}|P:[{periods}]}}"

def analyze_bitlayer(raw: bytes) -> dict:
    bits = bytes_to_bits(raw)
    signal = bits_to_signal(bits)
    sig = build_bit_signature(bits)
    return {"bits": bits[:128], "signal_preview": signal[:128], "signature": sig.to_dict(), "compact": compact_bit_expression(sig)}


def __main__():  # pragma: no cover
    print(analyze_bitlayer(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f"))

if __name__ == "__main__":  # pragma: no cover
    __main__()