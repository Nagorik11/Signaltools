"""Custom exceptions for signaltools."""

from __future__ import annotations


class SignalToolsError(Exception):
    """Base exception for the package."""


class SignalValidationError(SignalToolsError, ValueError):
    """Raised when an input signal or parameter is invalid."""
