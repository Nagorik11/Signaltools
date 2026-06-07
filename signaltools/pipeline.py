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
    
    # Note: For actual spectral analysis, you'd need to adapt these
    # to work in a pipeline context (they expect different signatures)
    
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
    
    # Add filtering if params provided
    if filter_params:
        from .filters import lowpass_filter
        pipeline.add_step(
            "lowpass",
            lowpass_filter,
            **filter_params
        )
    
    return pipeline


__all__ = [
    "AdvancedSignalAnalysis",
    "analyze_signal_advanced",
    "PipelineStep",
    "PipelineResult",
    "ProcessingPipeline",
    "create_analysis_pipeline",
    "create_preprocessing_pipeline"
]
