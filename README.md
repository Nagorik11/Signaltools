cd ..
# SignalTools

SignalTools is a lightweight Python library for digital signal processing, feature extraction, fingerprinting, binary data analysis, and layered symbolic summaries.

## What's improved in this polished build

- Fixed broken package imports and made `import signaltools as st` work correctly.
- Removed invalid code that was breaking `detect.py`.
- Added a real package entrypoint: `python -m signaltools ...`.
- Added advanced spectral tools:
  - STFT
  - spectrogram matrix
  - autocorrelation
  - pitch estimation
  - spectral centroid, bandwidth, and rolloff
  - band energy analysis
- Added advanced time-domain features:
  - MAD
  - standard deviation
  - waveform length
  - dynamic range
  - skewness
  - kurtosis
- Added adaptive event detection and onset strength.
- Added filtering utilities:
  - moving average
  - median filter
  - DC removal
  - FFT bandpass
- Added a formal fingerprint engine for signal comparison.
- Added an integrated high-level pipeline with `analyze_signal_advanced()`.
- Added `pyproject.toml` for easier packaging.

## Project Structure

```text
signaltools/
├── __init__.py
├── __main__.py
├── bitlayer.py
├── bridge.py
├── detect.py
├── features.py
├── filters.py
├── fingerprint.py
├── framing.py
├── modulate.py
├── pipeline.py
├── spectral.py
├── io/
│   └── __init__.py
├── docs/
├── samples/
└── test.py
```

## Installation

Local editable install:

```bash
pip install -e .
```

Or simply place the package folder inside your project and import it directly.

## Quick Start

### Basic analysis

```python
import signaltools as st

signal = [0.1, 0.5, -0.2, 0.8, -0.1, 0.4] * 100
analysis = st.analyze_signal_advanced(signal, sample_rate=1000)

print(analysis.summary)
print(analysis.spectral["centroid_hz"])
```

### Fingerprint generation

```python
import signaltools as st

signal_a = [0.0, 0.2, 0.7, -0.4, 0.1] * 100
signal_b = [0.0, 0.1, 0.6, -0.3, 0.2] * 100

fp_a = st.fingerprint_engine(signal_a, sample_rate=1000)
fp_b = st.fingerprint_engine(signal_b, sample_rate=1000)

print(st.compare_fingerprints(fp_a, fp_b))
```

### Spectrogram generation

```python
import signaltools as st

signal = [0.1, 0.5, -0.2, 0.8, -0.1, 0.4] * 200
spec = st.spectrogram_matrix(signal, frame_size=128, hop_size=64)
print(len(spec), len(spec[0]) if spec else 0)
```

### CLI usage

```bash
python -m signaltools samples/heartbeat.json --sample-rate 1000 --output output/report.json
```

## Main high-level APIs

- `analyze_signal_advanced(signal, sample_rate=...)`
- `fingerprint_engine(signal, sample_rate=...)`
- `compare_fingerprints(fp_a, fp_b)`
- `signal_signature(signal)`
- `analyze_signal_layered(signal)`
- `analyze_bitlayer(raw_bytes)`

## Run the demo test

```bash
python test.py
```


## Developer workflow

Install development tools:

```bash
pip install -e .[dev]
```

Run formatting and quality checks:

```bash
black .
ruff check .
pytest
```

Run benchmarks:

```bash
python benchmarks/run_benchmarks.py
```

## Forensic-ready usage

API rápida:

```python
import signaltools as st

result = st.forensic_analyze_signal(
    "samples/heartbeat.json",
    profile="generic_signal",
    case_id="CASE-001",
    examiner="Analyst",
    output_dir="output/forensic_bundle",
)
```

CLI:

```bash
python -m signaltools samples/heartbeat.json --forensic --profile generic_signal --case-id CASE-001 --examiner Analyst --bundle-dir output/forensic_bundle --output output/forensic_report.json
```

Esto genera:
- `manifest.json`
- `audit_trail.json`
- `analysis.json`
- `forensic_report.json`

## Image layer decomposition

```python
import signaltools as st

result = st.decompose_image_layers(
    image,
    background_kernel_size=15,
    reflection_percentile=98.0,
    foreground_mode="dark",
    wavelet_family="haar",
)
```

Capas principales:
- `background`
- `foreground`
- `reflections`
- `noise`
- `edges`
- `texture`

## Forensic image decomposition

```python
import signaltools as st

result = st.forensic_decompose_image(
    "input.png",
    case_id="IMG-001",
    examiner="Analyst",
    signer="Analyst",
    signing_key="secret",
    output_dir="output/image_bundle",
    background_kernel_size=15,
    reflection_percentile=98.0,
)
```

Esto genera un bundle con reporte, hashes por capa y las capas guardadas como imágenes.
