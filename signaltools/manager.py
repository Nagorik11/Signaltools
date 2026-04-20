"""Legacy manager wrapper around the core analysis classes."""

from __future__ import annotations

from .core.analyzer import SignalAnalyzer
from .core.signal import Signal


class Manager(Signal):
    """Convenience wrapper that owns a `SignalAnalyzer`."""

    def __init__(self, data: list[float]):
        super().__init__(data)
        self.analyzer = SignalAnalyzer(self)

    def summary(self) -> dict:
        """Return a summary of the managed signal."""
        return self.analyzer.generate_summary()


__all__ = ["Manager"]
