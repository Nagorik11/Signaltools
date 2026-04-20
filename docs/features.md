# Feature Extraction (features.py)

The `features` module implements standard time-domain feature extraction algorithms.

## Functions

### `mean(signal)`
Calculates the arithmetic mean of the signal.

### `rms(signal)`
Calculates the Root Mean Square (RMS) energy.

### `variance(signal)`
Calculates the statistical variance of the signal.

### `zero_crossing_rate(signal)`
Calculates the rate at which the signal crosses zero.

### `peak_to_peak(signal)`
Calculates the difference between the maximum and minimum values.

### `crest_factor(signal)`
Calculates the ratio of peak magnitude to RMS energy.

### `first_derivative(signal)` / `second_derivative(signal)`
Calculates the numerical derivatives of the signal.

### `frame_feature_vector(frame: list[float]) -> dict`
Returns a dictionary containing all basic features for a given frame.
