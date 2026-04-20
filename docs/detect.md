# Detection & Events (detect.py)

The `detect` module provides algorithms for identifying events and anomalies in signal data.

## Functions

### `threshold_events(signal: list[float], threshold: float) -> list[dict]`
Detects continuous regions where the signal's absolute magnitude exceeds a threshold.
Returns a list of event dictionaries with `start`, `end`, and `length`.

### `local_peaks(signal: list[float], min_height: float = 0.0) -> list[dict]`
Identifies local maxima in the signal that are greater than or equal to `min_height`.

### `anomaly_score(signal: list[float]) -> list[float]`
Calculates a Z-score based anomaly vector for the signal.
