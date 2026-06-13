"""High-level orchestration pipeline for advanced signal analysis.

This module provides both a fixed advanced analysis pipeline and a flexible
ProcessingPipeline class for building custom processing chains.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .bitlayer import analyze_bitlayer
from .bridge import analyze_signal_layered, signal_signature
from .detect import adaptive_events, anomaly_score, onset_strength
from .features import frame_feature_vector
from .fingerprint import fingerprint_engine
from .framing import FrameConfig, detrend_mean, frame_signal, normalize_signal
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class AdvancedSignalAnalysis:
    """Structured result returned by `analyze_signal_advanced`."""

    summary: dict[str, Any]
    frames: dict[str, Any]
    time_domain: dict[str, Any]
    spectral: dict[str, Any]
    temporal: dict[str, Any]
    symbolic: dict[str, Any]
    fingerprint: dict[str, Any]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass payload to a plain dictionary."""
        return asdict(self)


def analyze_signal_advanced(
    signal: list[float] | list[int],
    sample_rate: int = 44100,
    frame_size: int = 256,
    hop_size: int = 128,
) -> AdvancedSignalAnalysis:
    """Run the full advanced analysis pipeline over a 1D signal."""
    logger.debug(
        "Starting advanced analysis | sample_rate=%s frame_size=%s hop_size=%s",
        sample_rate,
        frame_size,
        hop_size,
    )
    prepared = detrend_mean(normalize_signal(signal))
    cfg = FrameConfig(frame_size=frame_size, hop_size=hop_size, pad_end=True, window="hann")
    frames = frame_signal(prepared, cfg)
    frame_features = [frame_feature_vector(frame) for frame in frames]

    signature = signal_signature(prepared, frame_size=frame_size, hop_size=hop_size)
    layered = analyze_signal_layered(prepared, source_type="numeric")
    fingerprint = fingerprint_engine(
        prepared,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
    )
    events = adaptive_events(prepared)
    anomalies = anomaly_score(prepared)
    onsets = onset_strength(prepared)
    
    from .spectral import (
        autocorrelation,
        estimate_pitch,
        spectral_bandwidth,
        spectral_centroid,
        spectral_rolloff,
        spectrogram_matrix,
    )
    
    specgram = spectrogram_matrix(prepared, frame_size=frame_size, hop_size=hop_size)
    ac = autocorrelation(prepared)

    bit_payload = np.asarray(prepared, dtype=np.float32).tobytes()
    bit_analysis = analyze_bitlayer(bit_payload)

    frame_aggregate: dict[str, float] = {}
    if frame_features:
        keys = frame_features[0].keys()
        frame_aggregate = {
            key: round(float(np.mean([row[key] for row in frame_features])), 6)
            for key in keys
        }

    result = AdvancedSignalAnalysis(
        summary={
            "samples": len(prepared),
            "frame_count": len(frames),
            "sample_rate": sample_rate,
        },
        frames={
            "config": {
                "frame_size": frame_size,
                "hop_size": hop_size,
                "window": cfg.window,
                "pad_end": cfg.pad_end,
            },
            "aggregate_features": frame_aggregate,
        },
        time_domain={
            "signature": signature.to_dict(),
            "autocorrelation_preview": [round(float(value), 6) for value in ac[:32]],
        },
        spectral={
            "centroid_hz": round(float(spectral_centroid(prepared, sample_rate=sample_rate)), 6),
            "bandwidth_hz": round(float(spectral_bandwidth(prepared, sample_rate=sample_rate)), 6),
            "rolloff_hz": round(float(spectral_rolloff(prepared, sample_rate=sample_rate)), 6),
            "pitch_hz": round(float(estimate_pitch(prepared, sample_rate=sample_rate)), 6),
            "spectrogram_shape": [len(specgram), len(specgram[0]) if specgram else 0],
            "spectrogram_preview": [
                [round(float(value), 4) for value in row[:8]] for row in specgram[:4]
            ],
        },
        temporal={
            "adaptive_event_count": len(events),
            "events_preview": events[:8],
            "onset_preview": [round(float(value), 6) for value in onsets[:32]],
        },
        symbolic={
            "layered": layered.to_dict(),
            "bitlayer": bit_analysis,
        },
        fingerprint=fingerprint.to_dict(),
        diagnostics={
            "max_anomaly_score": round(float(max(anomalies, default=0.0)), 6),
            "mean_anomaly_score": round(float(np.mean(anomalies)) if anomalies else 0.0, 6),
        },
    )
    logger.debug(
        "Completed advanced analysis | samples=%s frames=%s events=%s",
        len(prepared),
        len(frames),
        len(events),
    )
    return result


@dataclass
class PipelineStep:
    """Represents a single step in a processing pipeline."""
    
    name: str
    func: Callable[..., Any]
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    description: str = ""
    
    def execute(self, input_data: Any, context: dict[str, Any]) -> Any:
        """Execute this pipeline step.
        
        Args:
            input_data: Data from previous step (or initial input)
            context: Shared context dictionary across all steps
            
        Returns:
            Processed data to pass to next step
        """
        logger.info(f"Executing step: {self.name}")
        start_time = time.time()
        
        try:
            # Allow steps to access context via special kwarg
            exec_kwargs = self.kwargs.copy()
            if '_context' in exec_kwargs:
                exec_kwargs['_context'] = context
            
            result = self.func(input_data, *self.args, **exec_kwargs)
            
            elapsed = time.time() - start_time
            logger.debug(f"Step {self.name} completed in {elapsed:.4f}s")
            
            # Store metadata in context
            context.setdefault('step_metadata', {})[self.name] = {
                'execution_time': elapsed,
                'status': 'success'
            }
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Step {self.name} failed after {elapsed:.4f}s: {str(e)}")
            context.setdefault('step_metadata', {})[self.name] = {
                'execution_time': elapsed,
                'status': 'failed',
                'error': str(e)
            }
            raise


@dataclass
class PipelineResult:
    """Result of executing a ProcessingPipeline."""
    
    output: Any
    context: dict[str, Any]
    step_results: dict[str, Any]
    total_time: float
    success: bool
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'output_type': type(self.output).__name__,
            'context': self.context,
            'step_results': self.step_results,
            'total_time': self.total_time,
            'success': self.success
        }
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save pipeline result to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = self.to_dict()
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        save_data = convert_numpy(save_data)
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        logger.info(f"Pipeline result saved to {filepath}")


class ProcessingPipeline:
    """Flexible pipeline for chaining signal processing operations.
    
    This class allows building custom processing workflows by chaining
    multiple processing steps. Each step receives output from previous
    step and can share state via a context dictionary.
    
    Example:
        >>> from signaltools.filters import lowpass_filter
        >>> from signaltools.framing import normalize_signal
        >>> 
        >>> pipeline = ProcessingPipeline()
        >>> pipeline.add_step("normalize", normalize_signal)
        >>> pipeline.add_step("filter", lowpass_filter, cutoff=0.1)
        >>> result = pipeline.execute(signal_data)
    """
    
    def __init__(self, name: str = "unnamed_pipeline"):
        self.name = name
        self.steps: List[PipelineStep] = []
        self.logger = get_logger(__name__)
    
    def add_step(
        self,
        name: str,
        func: Callable[..., Any],
        *args,
        description: str = "",
        **kwargs
    ) -> 'ProcessingPipeline':
        """Add a processing step to the pipeline.
        
        Args:
            name: Unique identifier for this step
            func: Callable to execute
            *args: Positional arguments to pass to func
            description: Optional description of what this step does
            **kwargs: Keyword arguments to pass to func
            
        Returns:
            Self for method chaining
        """
        step = PipelineStep(
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            description=description
        )
        self.steps.append(step)
        self.logger.debug(f"Added step '{name}' to pipeline '{self.name}'")
        return self
    
    def insert_step(
        self,
        index: int,
        name: str,
        func: Callable[..., Any],
        *args,
        description: str = "",
        **kwargs
    ) -> 'ProcessingPipeline':
        """Insert a step at a specific position.
        
        Args:
            index: Position to insert (0-based)
            name: Unique identifier for this step
            func: Callable to execute
            *args: Positional arguments to pass to func
            description: Optional description
            **kwargs: Keyword arguments to pass to func
            
        Returns:
            Self for method chaining
        """
        step = PipelineStep(
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            description=description
        )
        self.steps.insert(index, step)
        return self
    
    def remove_step(self, name: str) -> 'ProcessingPipeline':
        """Remove a step by name.
        
        Args:
            name: Name of step to remove
            
        Returns:
            Self for method chaining
        """
        self.steps = [s for s in self.steps if s.name != name]
        return self
    
    def clear_steps(self) -> 'ProcessingPipeline':
        """Remove all steps from the pipeline."""
        self.steps = []
        return self
    
    def execute(
        self,
        input_data: Any,
        context: Optional[dict[str, Any]] = None,
        stop_on_error: bool = True
    ) -> PipelineResult:
        """Execute the entire pipeline.
        
        Args:
            input_data: Initial input data
            context: Optional shared context dictionary
            stop_on_error: If True, stop execution on first error
            
        Returns:
            PipelineResult with output and metadata
        """
        if context is None:
            context = {}
        
        context['pipeline_name'] = self.name
        context['start_time'] = time.time()
        context['step_metadata'] = {}
        
        self.logger.info(f"Starting pipeline '{self.name}' with {len(self.steps)} steps")
        start_time = time.time()
        
        current_data = input_data
        step_results = {}
        success = True
        
        for i, step in enumerate(self.steps):
            try:
                current_data = step.execute(current_data, context)
                step_results[step.name] = {
                    'status': 'success',
                    'output_type': type(current_data).__name__,
                    'output_shape': getattr(current_data, 'shape', None)
                }
            except Exception as e:
                step_results[step.name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                if stop_on_error:
                    success = False
                    break
                else:
                    self.logger.warning(f"Continuing despite step failure: {e}")
        
        total_time = time.time() - start_time
        context['total_time'] = total_time
        context['success'] = success
        
        self.logger.info(
            f"Pipeline '{self.name}' {'completed' if success else 'failed'} "
            f"in {total_time:.4f}s"
        )
        
        return PipelineResult(
            output=current_data,
            context=context,
            step_results=step_results,
            total_time=total_time,
            success=success
        )
    
    def execute_batch(
        self,
        inputs: List[Any],
        parallel: bool = False,
        max_workers: int = 4,
        context: Optional[dict[str, Any]] = None
    ) -> List[PipelineResult]:
        """Execute pipeline on multiple inputs.
        
        Args:
            inputs: List of input data items
            parallel: If True, use parallel execution
            max_workers: Maximum number of parallel workers
            context: Base context for all executions
            
        Returns:
            List of PipelineResult objects
        """
        if not parallel or len(inputs) <= 1:
            return [self.execute(inp, context=context.copy() if context else None) 
                    for inp in inputs]
        
        # Parallel execution using ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.execute, inp, context=context.copy() if context else None)
                for inp in inputs
            ]
            for future in futures:
                results.append(future.result())
        
        return results
    
    def to_config(self) -> dict[str, Any]:
        """Export pipeline configuration to dictionary.
        
        Note: Only steps with serializable parameters can be exported.
        Functions are represented by their qualified names.
        """
        config = {
            'name': self.name,
            'steps': []
        }
        
        for step in self.steps:
            step_config = {
                'name': step.name,
                'func': f"{step.func.__module__}.{step.func.__qualname__}",
                'args': step.args,
                'kwargs': step.kwargs,
                'description': step.description
            }
            config['steps'].append(step_config)
        
        return config
    
    def save_config(self, filepath: Union[str, Path]) -> None:
        """Save pipeline configuration to JSON file.
        
        Args:
            filepath: Path to save configuration
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        config = self.to_config()
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Pipeline config saved to {filepath}")
    
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'ProcessingPipeline':
        """Create pipeline from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured ProcessingPipeline instance
        """
        pipeline = cls(name=config.get('name', 'loaded_pipeline'))
        
        for step_config in config.get('steps', []):
            # Resolve function from string
            func_path = step_config['func']
            module_name, func_name = func_path.rsplit('.', 1)
            module = __import__(module_name, fromlist=[func_name])
            func = getattr(module, func_name)
            
            pipeline.add_step(
                name=step_config['name'],
                func=func,
                *step_config.get('args', []),
                description=step_config.get('description', ''),
                **step_config.get('kwargs', {})
            )
        
        return pipeline
    
    @classmethod
    def load_config(cls, filepath: Union[str, Path]) -> 'ProcessingPipeline':
        """Load pipeline from configuration file.
        
        Args:
            filepath: Path to configuration JSON file
            
        Returns:
            Configured ProcessingPipeline instance
        """
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        return cls.from_config(config)
    
    def __repr__(self) -> str:
        return f"ProcessingPipeline(name='{self.name}', steps={len(self.steps)})"
    
    def __len__(self) -> int:
        return len(self.steps)


# Convenience functions for common pipeline patterns

def create_analysis_pipeline(sample_rate: int = 44100) -> ProcessingPipeline:
    """Create a standard analysis pipeline.
    
    Args:
        sample_rate: Sample rate for spectral analysis
        
    Returns:
        Configured ProcessingPipeline
    """
    from .spectral import spectral_centroid, spectral_bandwidth
    
    pipeline = ProcessingPipeline(name="standard_analysis")
    
    pipeline.add_step(
        "normalize",
        normalize_signal,
        description="Normalize signal amplitude"
    )
    
    pipeline.add_step(
        "detrend",
        detrend_mean,
        description="Remove DC offset"
    )
    
    return pipeline


def create_preprocessing_pipeline(
    normalize: bool = True,
    detrend: bool = True,
    filter_params: Optional[dict] = None
) -> ProcessingPipeline:
    """Create a standard preprocessing pipeline.
    
    Args:
        normalize: Whether to normalize signal
        detrend: Whether to remove DC offset
        filter_params: Optional filter parameters
        
    Returns:
        Configured ProcessingPipeline
    """
    pipeline = ProcessingPipeline(name="preprocessing")
    
    if normalize:
        pipeline.add_step("normalize", normalize_signal)
    
    if detrend:
        pipeline.add_step("detrend", detrend_mean)
    
    if filter_params:
        from .filters import lowpass_filter
        pipeline.add_step(
            "lowpass",
            lowpass_filter,
            **filter_params
        )
    
    return pipeline


# === Spectral Analysis Pipelines ===

def _wrap_spectral_result(func, sample_rate: int, context_key: str):
    """Helper: execute a spectral function and store result in context."""
    from functools import wraps
    
    @wraps(func)
    def wrapper(signal, _context=None):
        result = func(signal, sample_rate=sample_rate)
        if _context is not None:
            _context[context_key] = result
        return signal
    return wrapper


def create_spectral_feature_pipeline(sample_rate: int = 44100) -> ProcessingPipeline:
    """Pipeline that extracts spectral features and stores them in context.
    
    Steps: normalize → detrend → centroid → bandwidth → rolloff → pitch
    
    Args:
        sample_rate: Sample rate in Hz
        
    Returns:
        Configured ProcessingPipeline
    """
    from .spectral import spectral_centroid, spectral_bandwidth, spectral_rolloff, estimate_pitch
    
    pipeline = ProcessingPipeline(name="spectral_features")
    
    pipeline.add_step("normalize", normalize_signal, description="Normalize amplitude")
    pipeline.add_step("detrend", detrend_mean, description="Remove DC offset")
    pipeline.add_step(
        "centroid", _wrap_spectral_result(spectral_centroid, sample_rate, "centroid_hz"),
        description="Spectral centroid",
        _context=None
    )
    pipeline.add_step(
        "bandwidth", _wrap_spectral_result(spectral_bandwidth, sample_rate, "bandwidth_hz"),
        description="Spectral bandwidth",
        _context=None
    )
    pipeline.add_step(
        "rolloff", _wrap_spectral_result(spectral_rolloff, sample_rate, "rolloff_hz"),
        description="Spectral rolloff",
        _context=None
    )
    pipeline.add_step(
        "pitch", _wrap_spectral_result(estimate_pitch, sample_rate, "pitch_hz"),
        description="Fundamental frequency estimation",
        _context=None
    )
    
    return pipeline


def create_spectrogram_pipeline(
    sample_rate: int = 44100,
    frame_size: int = 256,
    hop_size: int = 128,
    window: str = "hann",
    log_scale: bool = True,
) -> ProcessingPipeline:
    """Pipeline that computes a spectrogram from a signal.
    
    Steps: normalize → detrend → spectrogram (stored in context)
    
    Args:
        sample_rate: Sample rate in Hz
        frame_size: FFT frame size
        hop_size: Hop size between frames
        window: Window type ('hann', 'hamming', 'blackman', 'rectangular')
        log_scale: Apply dB scale if True
        
    Returns:
        Configured ProcessingPipeline
    """
    from .spectral import spectrogram_matrix
    
    pipeline = ProcessingPipeline(name="spectrogram")
    
    pipeline.add_step("normalize", normalize_signal, description="Normalize amplitude")
    pipeline.add_step("detrend", detrend_mean, description="Remove DC offset")
    
    def _compute_spectrogram(signal, _context=None):
        spec = spectrogram_matrix(
            signal,
            frame_size=frame_size,
            hop_size=hop_size,
            window=window,
            log_scale=log_scale,
        )
        if _context is not None:
            _context["spectrogram"] = spec
            _context["spectrogram_shape"] = [len(spec), len(spec[0]) if spec else 0]
        return spec
    
    pipeline.add_step(
        "spectrogram", _compute_spectrogram,
        description=f"Spectrogram ({frame_size} frame, {hop_size} hop)",
        _context=None
    )
    
    return pipeline


def create_band_energy_pipeline(
    sample_rate: int = 44100,
    bands: Optional[list[tuple[str, float, float]]] = None,
) -> ProcessingPipeline:
    """Pipeline that computes energy in multiple frequency bands.
    
    Default bands: sub-bass (0-60), bass (60-250), low-mid (250-500),
    mid (500-2000), high-mid (2000-4000), presence (4000-6000), brilliance (6000-20000)
    
    Steps: normalize → detrend → band energies (stored in context)
    
    Args:
        sample_rate: Sample rate in Hz
        bands: List of (name, low_hz, high_hz) tuples
        
    Returns:
        Configured ProcessingPipeline
    """
    from .spectral import band_energy
    
    if bands is None:
        bands = [
            ("sub_bass", 0.0, 60.0),
            ("bass", 60.0, 250.0),
            ("low_mid", 250.0, 500.0),
            ("mid", 500.0, 2000.0),
            ("high_mid", 2000.0, 4000.0),
            ("presence", 4000.0, 6000.0),
            ("brilliance", 6000.0, 20000.0),
        ]
    
    pipeline = ProcessingPipeline(name="band_energy")
    
    pipeline.add_step("normalize", normalize_signal, description="Normalize amplitude")
    pipeline.add_step("detrend", detrend_mean, description="Remove DC offset")
    
    def _compute_band_energies(signal, _context=None):
        energies = {}
        for name, low, high in bands:
            energies[name] = round(band_energy(signal, sample_rate, low, high), 6)
        if _context is not None:
            _context["band_energies"] = energies
        return signal
    
    pipeline.add_step(
        "band_energies", _compute_band_energies,
        description=f"Energy in {len(bands)} frequency bands",
        _context=None
    )
    
    return pipeline


def create_full_spectral_pipeline(sample_rate: int = 44100) -> ProcessingPipeline:
    """Complete spectral analysis pipeline combining features, spectrogram, and band energy.
    
    Stores all results in context under 'spectral_features', 'spectrogram', and 'band_energies'.
    Final output is the spectrogram matrix.
    
    Args:
        sample_rate: Sample rate in Hz
        
    Returns:
        Configured ProcessingPipeline
    """
    from .spectral import (
        spectral_centroid, spectral_bandwidth, spectral_rolloff,
        estimate_pitch, band_energy, spectrogram_matrix,
    )
    
    pipeline = ProcessingPipeline(name="full_spectral_analysis")
    
    pipeline.add_step("normalize", normalize_signal, description="Normalize amplitude")
    pipeline.add_step("detrend", detrend_mean, description="Remove DC offset")
    
    # Single spectral feature extraction step
    def _compute_features(signal, _context=None):
        features = {
            "centroid_hz": round(spectral_centroid(signal, sample_rate=sample_rate), 6),
            "bandwidth_hz": round(spectral_bandwidth(signal, sample_rate=sample_rate), 6),
            "rolloff_hz": round(spectral_rolloff(signal, sample_rate=sample_rate), 6),
            "pitch_hz": round(estimate_pitch(signal, sample_rate=sample_rate), 6),
        }
        if _context is not None:
            _context["spectral_features"] = features
        return signal
    
    pipeline.add_step(
        "features", _compute_features,
        description="Spectral centroid, bandwidth, rolloff, pitch",
        _context=None
    )
    
    # Band energy
    bands = [
        ("sub_bass", 0.0, 60.0),
        ("bass", 60.0, 250.0),
        ("low_mid", 250.0, 500.0),
        ("mid", 500.0, 2000.0),
        ("high_mid", 2000.0, 4000.0),
        ("presence", 4000.0, 6000.0),
        ("brilliance", 6000.0, 20000.0),
    ]
    
    def _compute_band_energies(signal, _context=None):
        energies = {}
        for name, low, high in bands:
            energies[name] = round(band_energy(signal, sample_rate, low, high), 6)
        if _context is not None:
            _context["band_energies"] = energies
        return signal
    
    pipeline.add_step(
        "band_energies", _compute_band_energies,
        description=f"Energy in {len(bands)} bands",
        _context=None
    )
    
    # Spectrogram
    def _compute_spectrogram(signal, _context=None):
        spec = spectrogram_matrix(signal, frame_size=256, hop_size=128)
        if _context is not None:
            _context["spectrogram"] = spec
            _context["spectrogram_shape"] = [len(spec), len(spec[0]) if spec else 0]
        return spec
    
    pipeline.add_step(
        "spectrogram", _compute_spectrogram,
        description="Spectrogram (256 frame, 128 hop)",
        _context=None
    )
    
    # Aggregate summary
    def _aggregate(signal, _context=None):
        if _context is None:
            return signal
        features = _context.get("spectral_features", {})
        _context["summary"] = {
            **features,
            "band_energies": _context.get("band_energies"),
            "spectrogram_shape": _context.get("spectrogram_shape"),
        }
        return signal
    
    pipeline.add_step("aggregate", _aggregate, description="Aggregate spectral summary", _context=None)
    
    return pipeline


def create_stft_analysis_pipeline(
    sample_rate: int = 44100,
    frame_size: int = 256,
    hop_size: int = 128,
) -> ProcessingPipeline:
    """Pipeline for STFT-based time-frequency analysis.
    
    Steps: normalize → detrend → STFT → spectral features per frame (stored in context)
    Final output is the STFT magnitude matrix.
    
    Args:
        sample_rate: Sample rate in Hz
        frame_size: FFT frame size
        hop_size: Hop size between frames
        
    Returns:
        Configured ProcessingPipeline
    """
    from .spectral import stft, spectral_centroid
    
    pipeline = ProcessingPipeline(name="stft_analysis")
    
    pipeline.add_step("normalize", normalize_signal, description="Normalize amplitude")
    pipeline.add_step("detrend", detrend_mean, description="Remove DC offset")
    
    def _compute_stft(signal, _context=None):
        spec = stft(signal, frame_size=frame_size, hop_size=hop_size)
        if _context is not None:
            _context["stft_matrix"] = spec
            _context["stft_shape"] = [len(spec), len(spec[0]) if spec else 0]
        return spec
    
    pipeline.add_step(
        "stft", _compute_stft,
        description=f"STFT ({frame_size} frame, {hop_size} hop)",
        _context=None
    )
    
    return pipeline


def create_metrics_pipeline(sample_rate: int = 44100) -> ProcessingPipeline:
    """Pipeline that computes comprehensive signal metrics.

    Computes: RMS, Energy, Power, SNR, Dominant Frequency, Bandwidth,
    PSD, Variance, Autocorrelation, Spectral Entropy.

    All metrics stored in context under 'metrics'. Final output is
    the preprocessed signal (pass-through).

    Args:
        sample_rate: Sample rate in Hz

    Returns:
        Configured ProcessingPipeline
    """
    from .spectral import (
        dominant_frequency, spectral_bandwidth, power_spectral_density,
        spectral_entropy, estimate_snr, autocorrelation,
    )
    from .features import rms, signal_energy, signal_power, variance

    pipeline = ProcessingPipeline(name="metrics")

    pipeline.add_step("normalize", normalize_signal, description="Normalize amplitude")
    pipeline.add_step("detrend", detrend_mean, description="Remove DC offset")

    def _compute_metrics(signal, _context=None):
        signal_len = len(signal)
        metrics = {
            "rms": round(rms(signal), 6),
            "energy": round(signal_energy(signal), 6),
            "power": round(signal_power(signal), 6),
            "variance": round(variance(signal), 6),
            "snr_db": estimate_snr(signal),
            "dominant_freq_hz": round(dominant_frequency(signal, sample_rate), 6),
            "bandwidth_hz": round(spectral_bandwidth(signal, sample_rate), 6),
            "spectral_entropy": round(spectral_entropy(signal), 6),
            "n_samples": signal_len,
            "sample_rate": sample_rate,
        }
        freqs, psd = power_spectral_density(signal, sample_rate)
        metrics["psd_freqs"] = [round(f, 4) for f in freqs]
        metrics["psd_values"] = [round(p, 10) for p in psd]
        metrics["psd_peak"] = round(max(psd), 10)

        ac = autocorrelation(signal, normalize=True)
        metrics["autocorr"] = [round(v, 6) for v in ac[:32]]

        if _context is not None:
            _context["metrics"] = metrics
        return signal

    pipeline.add_step(
        "metrics", _compute_metrics,
        description="RMS, Energy, Power, SNR, Dominant Freq, Bandwidth, PSD, Variance, Autocorrelation, Entropy",
        _context=None
    )

    return pipeline


__all__ = [
    "AdvancedSignalAnalysis",
    "analyze_signal_advanced",
    "PipelineStep",
    "PipelineResult",
    "ProcessingPipeline",
    "create_analysis_pipeline",
    "create_preprocessing_pipeline",
    "create_spectral_feature_pipeline",
    "create_spectrogram_pipeline",
    "create_band_energy_pipeline",
    "create_full_spectral_pipeline",
    "create_stft_analysis_pipeline",
    "create_metrics_pipeline",
]
