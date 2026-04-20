# Bridge & Signature (bridge.py)

The `bridge` module connects low-level features to high-level descriptors and representations.

## Classes

### `SignalSignature`
A comprehensive dataclass capturing the entire state of a signal analysis.
Includes dimensions, mean frame features, spectral properties, event counts, and derivatives.

## Functions

### `signal_signature(...) -> SignalSignature`
Generates a full `SignalSignature` from a raw signal. This is the primary high-level API for signal characterization.

### `signature_to_glyph_vector(sig: SignalSignature) -> list[float]`
Compresses a `SignalSignature` into a flat list of floats, ideal for similarity matching, clustering, or visualization.
