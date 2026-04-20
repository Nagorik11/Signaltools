"""Signal modulation helpers."""

from __future__ import annotations

import math

from .utils import ensure_positive_int, to_1d_float_array


def amplitude_modulation(carrier: list[float], modulator: list[float]) -> list[float]:
    """Apply amplitude modulation using a normalized modulator.

    The result is computed as::

        output[n] = carrier[n] * (1 + modulator[n])
    """
    carrier_arr = to_1d_float_array(carrier, name="carrier")
    modulator_arr = to_1d_float_array(modulator, name="modulator")
    n = min(len(carrier_arr), len(modulator_arr))
    return (carrier_arr[:n] * (1.0 + modulator_arr[:n])).tolist()


def frequency_modulation(carrier_freq: float, modulator: list[float], sample_rate: int = 44100, index: float = 1.0) -> list[float]:
    """Generate an FM signal from a modulator waveform."""
    sample_rate = ensure_positive_int(sample_rate, "sample_rate")
    carrier_freq = float(carrier_freq)
    index = float(index)
    modulator_arr = to_1d_float_array(modulator, name="modulator")

    signal: list[float] = []
    phase = 0.0
    for sample in modulator_arr:
        freq = carrier_freq + (index * float(sample))
        phase += 2.0 * math.pi * freq / sample_rate
        phase %= 2.0 * math.pi
        signal.append(math.sin(phase))
    return signal


__all__ = ["amplitude_modulation", "frequency_modulation"]
