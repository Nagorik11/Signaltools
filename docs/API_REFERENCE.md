# SignalTools API Reference

## Package entrypoint

```python
import signaltools as st
```

## Logging

### `configure_logging(level="INFO")`
Configures package logging for CLI and library usage.

### `get_logger(name="signaltools")`
Returns a namespaced logger.

## Ingestion and I/O

### `read_signal_file(path)`
Reads a file into a `SignalBuffer`.

### `guess_numeric_views(raw)`
Returns decoded views such as `uint8`, `int16_le`, `uint16_le`, and `float32_le`.

### `write_wav(path, signal, sample_rate=44100)`
Writes a mono 16-bit PCM WAV.

### `read_wav(path)`
Reads a WAV file into a float list.

### `read_audio_file(path, sample_rate=44100, filters=None)`
Uses FFmpeg to decode audio into a raw mono PCM buffer.

### `Ingestor`
Helpers for loading signals from:
- WAV
- JSON
- text
- PCAP
- video

## Framing and normalization

### `FrameConfig`
Dataclass with:
- `frame_size`
- `hop_size`
- `pad_end`
- `window`

### `frame_signal(signal, cfg)`
Splits a 1D signal into frames.

### `normalize_signal(signal)`
Peak-normalizes a signal.

### `detrend_mean(signal)`
Subtracts the mean from a signal.

### `standardize_signal(signal)`
Applies z-score standardization.

## Time-domain features

### `frame_feature_vector(frame)`
Returns a dictionary with features such as:
- mean
- median
- variance
- stddev
- MAD
- RMS
- energy
- ZCR
- peak-to-peak
- crest factor
- waveform length
- dynamic range
- skewness
- kurtosis

### Derivatives
- `first_derivative(signal)`
- `second_derivative(signal)`

## Spectral analysis

### `dft(signal, mode="fft")`
Returns real, imaginary, magnitude, and phase components.

### `dominant_bins(magnitude, top_k=8)`
Returns the strongest frequency bins.

### `spectral_energy(magnitude)`
Computes total spectral energy.

### `spectral_flatness(magnitude)`
Computes spectral flatness.

### `frequency_axis(n_samples, sample_rate, real_only=True)`
Returns the matching frequency axis.

### `power_spectrum(signal)`
Returns the real-valued power spectrum.

### Spectral descriptors
- `spectral_centroid(signal, sample_rate=44100)`
- `spectral_bandwidth(signal, sample_rate=44100)`
- `spectral_rolloff(signal, sample_rate=44100, roll_percent=0.85)`
- `band_energy(signal, sample_rate, low_hz, high_hz)`

### Time-frequency
- `stft(signal, frame_size=256, hop_size=128, window="hann")`
- `spectrogram_matrix(signal, frame_size=256, hop_size=128, window="hann", log_scale=True)`

### Periodicity
- `autocorrelation(signal, normalize=True)`
- `estimate_pitch(signal, sample_rate=44100, min_hz=50.0, max_hz=1000.0)`

## Detection

### `threshold_events(signal, threshold)`
Finds contiguous regions over a fixed threshold.

### `adaptive_threshold(signal, z=2.0)`
Builds a threshold from the signal amplitude distribution.

### `adaptive_events(signal, z=2.0)`
Event detection using the adaptive threshold.

### `local_peaks(signal, min_height=0.0, min_distance=1)`
Returns local maxima.

### `anomaly_score(signal)`
Returns per-sample absolute z-score anomaly levels.

### `onset_strength(signal)`
Returns a simple onset envelope.

## Filters

### `moving_average(signal, window_size=5)`
Centered moving average filter.

### `median_filter(signal, window_size=5)`
Median filter with edge padding.

### `remove_dc(signal)`
Subtracts the DC component.

### `normalize_peak(signal)`
Peak normalization helper.

### `fft_bandpass(signal, sample_rate, low_hz, high_hz)`
Bandpass using FFT masking.

## Bitlayer and symbolic analysis

### `analyze_bitlayer(raw)`
Returns a binary-layer preview and compact summary.

### `build_bit_signature(bits)`
Builds a `BitSignature` dataclass.

### `signal_signature(signal, frame_size=256, hop_size=128, threshold=0.35)`
High-level signature over a normalized signal.

### `analyze_signal_layered(signal, source_type="unknown")`
Produces a layered symbolic analysis structure.

### `reconstruct_signal_from_signature(sig, duration=1.0, sample_rate=44100)`
Rebuilds an approximate waveform from a signature.

## Fingerprints

### `SignalFingerprint`
Dataclass with `vector`, `labels`, and `meta`.

### `fingerprint_engine(signal, sample_rate=44100, frame_size=256, hop_size=128)`
Produces a comparison-ready fingerprint vector.

### Comparison helpers
- `cosine_similarity(a, b)`
- `euclidean_distance(a, b)`
- `compare_fingerprints(fp_a, fp_b)`

## High-level pipeline

### `AdvancedSignalAnalysis`
Dataclass with:
- `summary`
- `frames`
- `time_domain`
- `spectral`
- `temporal`
- `symbolic`
- `fingerprint`
- `diagnostics`

### `analyze_signal_advanced(signal, sample_rate=44100, frame_size=256, hop_size=128)`
Runs the integrated advanced analysis pipeline.

## Modulation

### `amplitude_modulation(carrier, modulator)`
Applies AM using `carrier * (1 + modulator)`.

### `frequency_modulation(carrier_freq, modulator, sample_rate=44100, index=1.0)`
Generates an FM waveform.

## Exceptions

### `SignalToolsError`
Base package exception.

### `SignalValidationError`
Raised when invalid input or parameters are detected.

## Quality tooling

### Ruff
```bash
ruff check .
```

### Black
```bash
black .
```

### Pytest
```bash
pytest -q
```

### Coverage
```bash
pytest --cov=signaltools --cov-report=term-missing
```

### Benchmarks
```bash
python benchmarks/run_benchmarks.py
```

## Filter design module

### FIR design
- `fir_lowpass(num_taps, cutoff_hz, sample_rate, window="hamming")`
- `fir_highpass(num_taps, cutoff_hz, sample_rate, window="hamming")`
- `fir_bandpass(num_taps, low_hz, high_hz, sample_rate, window="hamming")`
- `fir_bandstop(num_taps, low_hz, high_hz, sample_rate, window="hamming")`
- `fractional_delay_fir(num_taps, delay_samples, window="hamming")`
- `differentiator_fir()`

### IIR design
- `iir_lowpass_single_pole(cutoff_hz, sample_rate)`
- `iir_highpass_single_pole(cutoff_hz, sample_rate)`
- `iir_integrator_leaky(alpha=0.99)`

### Biquads
- `biquad_lowpass(cutoff_hz, sample_rate, q=0.707...)`
- `biquad_highpass(cutoff_hz, sample_rate, q=0.707...)`
- `biquad_bandpass(center_hz, sample_rate, q=1.0)`
- `biquad_notch(center_hz, sample_rate, q=30.0)`
- `biquad_allpass(center_hz, sample_rate, q=0.707)`

### Special structures
- `comb_filter_feedforward(delay_samples, gain=0.5)`
- `comb_filter_feedback(delay_samples, gain=0.5)`

### Smoothing / envelope / adaptive
- `savitzky_golay_coefficients(window_length, polyorder, deriv=0)`
- `savitzky_golay_filter(signal, window_length, polyorder, deriv=0)`
- `hilbert_transform_fft(signal)`
- `analytic_signal(signal)`
- `envelope(signal)`
- `lms_adaptive_filter(desired, reference, num_taps=8, step_size=0.01)`

### Application helpers
- `apply_fir(signal, coeffs)`
- `apply_iir(signal, coeffs)`

## Kalman / Wiener
- `kalman_filter_1d(signal, process_variance=..., measurement_variance=...)`
- `wiener_filter_1d(signal, window_size=5, noise_variance=None)`

## Morfológicos / mediana avanzada
- `advanced_median_filter(signal, window_size=5)`
- `rank_filter(signal, window_size=5, rank=0)`
- `dilation_1d(signal, window_size=3)`
- `erosion_1d(signal, window_size=3)`
- `opening_1d(signal, window_size=3)`
- `closing_1d(signal, window_size=3)`
- `morphological_gradient_1d(signal, window_size=3)`

## Grafos
- `graph_laplacian(adjacency, normalized=True)`
- `graph_fourier_basis(adjacency, normalized=True)`
- `graph_filter_signal(signal, adjacency, response=None)`
- `graph_polynomial_filter(signal, adjacency, coeffs)`

## Multirate / polifásico
- `polyphase_decompose(coeffs, phases)`
- `decimate(signal, factor, fir_taps=31)`
- `interpolate(signal, factor, fir_taps=31)`
- `two_band_analysis_bank(signal, fir_taps=31)`

## EKF / UKF / Particle / Wiener adaptativo
- `adaptive_wiener_filter_1d(signal, window_size=5, adaptation_rate=0.1)`
- `extended_kalman_filter(measurements, transition_fn, measurement_fn, transition_jacobian, measurement_jacobian, ...)`
- `unscented_kalman_filter(measurements, transition_fn, measurement_fn, ...)`
- `particle_filter_1d(measurements, num_particles=100, process_std=0.1, measurement_std=0.2)`

## Morfología 2D
- `dilation_2d(image, kernel_size=3)`
- `erosion_2d(image, kernel_size=3)`
- `opening_2d(image, kernel_size=3)`
- `closing_2d(image, kernel_size=3)`
- `median_filter_2d(image, kernel_size=3)`
- `morphological_gradient_2d(image, kernel_size=3)`

## Bancos multibanda
- `haar_analysis_bank(signal)`
- `haar_synthesis_bank(low, high)`
- `uniform_filter_bank(signal, bands=4, fir_taps=31)`
- `reconstruct_uniform_filter_bank(subbands, bands=4, fir_taps=31)`

## Graph wavelets / Chebyshev
- `chebyshev_graph_filter(signal, adjacency, coeffs)`
- `graph_wavelet_kernel(lambdas, scale, kind="heat")`
- `graph_wavelet_transform(signal, adjacency, scales, kind="heat")`

## RTS / smoother backward / partículas no lineales
- `backward_exponential_smoother(signal, alpha=0.3)`
- `rts_smoother(filtered_estimates, filtered_covariances, transition_matrix, process_covariance)`
- `particle_filter_nonlinear(measurements, transition_fn, likelihood_fn, initial_particles, ...)`

## Morfología 3D
- `dilation_3d(volume, kernel_size=3)`
- `erosion_3d(volume, kernel_size=3)`
- `opening_3d(volume, kernel_size=3)`
- `closing_3d(volume, kernel_size=3)`
- `median_filter_3d(volume, kernel_size=3)`
- `morphological_gradient_3d(volume, kernel_size=3)`

## Wavelet packet
- `wavelet_packet_decompose(signal, level=3)`
- `wavelet_packet_reconstruct(tree)`

## Graph scattering / spectral GNN
- `spectral_gnn_filter(signal, adjacency, weights, activation="relu")`
- `graph_scattering_transform(signal, adjacency, scales)`

- `particle_filter_multivariate(measurements, transition_fn, likelihood_fn, initial_particles, ...)`

## Morfología volumétrica con kernels arbitrarios
- `dilation_3d_kernel(volume, kernel)`
- `erosion_3d_kernel(volume, kernel)`
- `opening_3d_kernel(volume, kernel)`
- `closing_3d_kernel(volume, kernel)`

## Wavelet packet multifámilia
- `wavelet_filters(family)`
- `wavelet_packet_decompose(signal, level=3, family="haar")`
- `wavelet_packet_reconstruct(tree)`

## GNN apilable con pooling y residuals
- `graph_pool(signal, factor=2, mode="mean")`
- `stacked_gnn(signal, adjacency, layer_weights, activation="relu", pooling_factor=1, pooling_mode="mean", residual=False)`


## Familias wavelet extendidas
- `available_wavelet_families()`
- `wavelet_filters(family)`
- familias soportadas: `haar`, `db2`, `db4`, `db6`, `sym2`, `sym4`, `coif1`, `coif2`

## GNN profundo con normalización y attention
- `graph_attention_matrix(signal, adjacency, alpha=1.0, add_self_loops=True)`
- `graph_attention_filter(signal, adjacency, alpha=1.0, activation="linear")`
- `deep_gnn_stack(signal, adjacency, layer_weights, activation="relu", normalization="layernorm", attention=False, attention_alpha=1.0, attention_mix=0.5, pooling_factor=1, pooling_mode="mean", residual=False)`


## Wavelets ortogonales y biortogonales adicionales
- `wavelet_family_kind(family)`
- nuevas familias ortogonales: `db8`, `sym6`, `coif3`
- nuevas familias biortogonales: `bior53`, `bior97`

## Attention multi-head y normalización formal por bloque
- `graph_block_normalize(signal, mode="layernorm", gamma=1.0, beta=0.0, eps=1e-5)`
- `multihead_graph_attention(signal, adjacency, num_heads=4, alpha=1.0, value_scales=None, concat=False, add_self_loops=True, activation="linear")`
- `deep_gnn_stack(..., attention=True, num_heads=4, normalization="batchnorm" | "layernorm", norm_gamma=1.0, norm_beta=0.0, norm_eps=1e-5)`


## Graph transformers simplificados y multicanal por nodo
- `graph_block_normalize_multichannel(features, mode="layernorm", gamma=1.0, beta=0.0, eps=1e-5)`
- `multihead_graph_attention_multichannel(features, adjacency, num_heads=4, alpha=1.0, concat=False, add_self_loops=True, activation="linear")`
- `spectral_gnn_filter_multichannel(features, adjacency, weights, normalized=True, activation="relu")`
- `graph_transformer_layer(features, adjacency, num_heads=4, attention_alpha=1.0, normalization="layernorm", activation="gelu", ff_gain=2.0, residual=True)`
- `graph_transformer_stack(features, adjacency, depth=2, num_heads=4, attention_alpha=1.0, normalization="layernorm", activation="gelu", ff_gain=2.0, residual=True)`

## Familias spline/CDF y biortogonales ampliadas
- nuevas familias: `cdf53`, `cdf97`, `spline53`, `spline97`, `bior22`, `bior44`


## Message passing con edge features y transformer Q/K/V explícito
- `edge_aware_message_passing(node_features, adjacency, edge_features, self_weight=1.0, neighbor_weight=1.0, aggregation="mean", activation="linear")`
- `edge_feature_message_passing_stack(node_features, adjacency, edge_features, depth=2, ...)`
- `qkv_graph_attention(node_features, adjacency, num_heads=4, query_scales=None, key_scales=None, value_scales=None, concat=False, add_self_loops=True, activation="linear", edge_features=None)`
- `graph_transformer_qkv_layer(node_features, adjacency, num_heads=4, ..., edge_features=None)`
- `graph_transformer_qkv_stack(node_features, adjacency, depth=2, ..., edge_features=None)`

## Wavelet packets 2D para imágenes
- `WaveletPacket2DTree`
- `wavelet_packet_2d_decompose(image, level=2, family="haar")`
- `wavelet_packet_2d_reconstruct(tree)`


## Edge-conditioned convolutions y transformer con masking
- `EdgeConditionedConvResult`
- `edge_conditioned_convolution(node_features, adjacency, edge_features, self_weight=1.0, neighbor_weight=1.0, aggregation="mean", activation="linear")`
- `edge_conditioned_conv_stack(node_features, adjacency, edge_features, depth=2, ...)`
- `masked_qkv_graph_attention(node_features, adjacency, num_heads=4, ..., edge_features=None, attention_mask=None, edge_bias_scale=1.0)`
- `graph_transformer_masked_qkv_layer(node_features, adjacency, ..., edge_features=None, attention_mask=None, edge_bias_scale=1.0)`
- `graph_transformer_masked_qkv_stack(node_features, adjacency, depth=2, ..., edge_features=None, attention_mask=None, edge_bias_scale=1.0)`

## Wavelet packets 3D para volúmenes
- `WaveletPacket3DTree`
- `wavelet_packet_3d_decompose(volume, level=2, family="haar")`
- `wavelet_packet_3d_reconstruct(tree)`


## Channel mixing, positional encodings y wavelet packet anisotrópico 3D
- `ChannelMixResult`
- `channel_mix(node_features, mix_matrix=None, bias=None, activation="linear", residual=False, out_channels=None, mix_strength=0.15)`
- `laplacian_positional_encoding(adjacency, dimensions=4, normalized=True)`
- `random_walk_positional_encoding(adjacency, steps=4)`
- `augment_with_graph_positional_encoding(node_features, adjacency, method="laplacian", dimensions=4, steps=4)`
- `graph_transformer_enhanced_layer(..., positional_encoding_method=None, positional_dimensions=4, positional_steps=4, channel_mix_matrix=None, channel_mix_bias=None, mix_strength=0.15)`
- `graph_transformer_enhanced_stack(...)`
- `AnisotropicWaveletPacket3DTree`
- `anisotropic_wavelet_packet_3d_decompose(volume, level=2, families=("haar", "haar", "haar"))`
- `anisotropic_wavelet_packet_3d_reconstruct(tree)`


## Multicanal complejo/analítico, edge embeddings y wavelet packet 4D
- `ComplexChannelResult`
- `analytic_signal_multichannel(features)`
- `complex_channel_mix(features, mix_matrix=None, bias=None, residual=False, out_channels=None, mix_strength=0.15)`
- `structured_edge_embedding_attention(node_features, adjacency, edge_embeddings, num_heads=4, concat=False, activation="linear", add_self_loops=True, edge_embedding_scale=1.0)`
- `graph_transformer_edge_embedding_layer(node_features, adjacency, edge_embeddings, ...)`
- `graph_transformer_edge_embedding_stack(node_features, adjacency, edge_embeddings, depth=2, ...)`
- `WaveletPacket4DTree`
- `wavelet_packet_4d_decompose(tensor, level=1, family="haar")`
- `wavelet_packet_4d_reconstruct(tree)`


## Operadores espectrales complejos, attention nodo-arista-temporal y packet 4D anisotrópico
- `ComplexSpectralResult`
- `complex_dft_multichannel(features)`
- `complex_spectral_mask(features, mask=None, phase_shift=0.0)`
- `complex_spectral_shift(features, bins=1)`
- `HybridTemporalAttentionResult`
- `hybrid_node_edge_temporal_attention(sequence, adjacency, edge_embeddings, num_heads=4, temporal_window=1, activation="linear", edge_embedding_scale=1.0, temporal_decay=1.0)`
- `hybrid_graph_temporal_transformer_layer(sequence, adjacency, edge_embeddings, ...)`
- `hybrid_graph_temporal_transformer_stack(sequence, adjacency, edge_embeddings, depth=2, ...)`
- `AnisotropicWaveletPacket4DTree`
- `anisotropic_wavelet_packet_4d_decompose(tensor, level=1, families=("haar", "haar", "haar", "haar"))`
- `anisotropic_wavelet_packet_4d_reconstruct(tree)`


## Complex frame-domain operators

- `ComplexFrameResult`
- `complex_stft_multichannel(...)`
- `complex_frame_operator(...)`

## Recurrent hybrid graph-temporal attention

- `RecurrentHybridAttentionResult`
- `recurrent_hybrid_node_edge_temporal_attention(...)`
- `recurrent_hybrid_graph_temporal_transformer_stack(...)`

## 5D wavelet packets

- `WaveletPacket5DTree`
- `AnisotropicWaveletPacket5DTree`
- `wavelet_packet_5d_decompose(...)`
- `wavelet_packet_5d_reconstruct(...)`
- `anisotropic_wavelet_packet_5d_decompose(...)`
- `anisotropic_wavelet_packet_5d_reconstruct(...)`


## Learnable-style complex time-frequency operators

- `ComplexLearnableTFResult`
- `complex_learnable_tf_operator(...)`
- `complex_learnable_tf_stack(...)`

## Gated recurrent hybrid memory

- `HybridGatedMemoryResult`
- `hybrid_graph_temporal_gated_memory(...)`
- `hybrid_graph_temporal_gated_stack(...)`

## Rich anisotropic 5D wavelet packets

- `WaveletPacket5DTree`
- `AnisotropicWaveletPacket5DTree`
- richer axis-wise families such as `db8`, `coif3`, `bior97`, `cdf97`, `sym6`


## Multi-head complex time-frequency operators

- `ComplexMultiHeadTFResult`
- `multihead_band_complex_tf_operator(...)`
- `multihead_band_complex_tf_stack(...)`

## Bidirectional gated memory

- `BidirectionalGatedMemoryResult`
- `bidirectional_hybrid_graph_temporal_gated_memory(...)`
- `bidirectional_hybrid_graph_temporal_gated_stack(...)`

## Adaptive anisotropic 5D wavelet packets

- `AdaptiveWaveletPacket5DTree`
- `select_wavelet_family_per_axis_5d(...)`
- `adaptive_wavelet_packet_5d_decompose(...)`
- `adaptive_wavelet_packet_5d_reconstruct(...)`


## Complex multiband head coupling

- `ComplexCoupledAttentionTFResult`
- `complex_multiband_head_coupling_operator(...)`
- `complex_multiband_head_coupling_stack(...)`

## Hierarchical gated memory

- `HierarchicalGatedMemoryResult`
- `hierarchical_hybrid_graph_temporal_gated_memory(...)`
- `hierarchical_hybrid_graph_temporal_gated_stack(...)`

## Local block-guided adaptive 5D packets

- `BlockAdaptiveWaveletPacket5DTree`
- `select_wavelet_family_per_axis_5d_local_blocks(...)`
- `adaptive_blockwise_wavelet_packet_5d_decompose(...)`
- `adaptive_blockwise_wavelet_packet_5d_reconstruct(...)`


## Time-dependent complex coupling

- `temporal_complex_head_coupling_operator(...)`

## Multiscale hierarchical gating

- `MultiscaleHierarchicalGatedMemoryResult`
- `multiscale_hierarchical_hybrid_graph_temporal_gated_memory(...)`
- `multiscale_hierarchical_hybrid_graph_temporal_gated_stack(...)`

## Spatially variable packet trees

- `SpatiallyAdaptiveWaveletPacket5DTree`
- `spatially_variable_wavelet_packet_5d_decompose(...)`
- `spatially_variable_wavelet_packet_5d_reconstruct(...)`


## Content-conditioned temporal coupling

- `content_conditioned_temporal_head_coupling_operator(...)`

## Attentive multiscale hierarchical gating

- `AttentiveMultiscaleHierarchicalGatedMemoryResult`
- `attentive_multiscale_hierarchical_hybrid_graph_temporal_gated_memory(...)`
- `attentive_multiscale_hierarchical_hybrid_graph_temporal_gated_stack(...)`

## Subband-adaptive packet trees

- `SubbandAdaptiveWaveletPacket5DTree`
- `subband_adaptive_wavelet_packet_5d_decompose(...)`
- `subband_adaptive_wavelet_packet_5d_reconstruct(...)`


## Causal / noncausal temporal coupling

- `mode_conditioned_temporal_head_coupling_operator(...)`

## Regularized adaptive selection

- `RegularizedAdaptiveWaveletPacket5DTree`
- `regularized_select_wavelet_family_per_axis_5d(...)`
- `regularized_adaptive_wavelet_packet_5d_decompose(...)`
- `regularized_adaptive_wavelet_packet_5d_reconstruct(...)`

## Attentive subband packet trees

- `SubbandAttentiveWaveletPacket5DTree`
- `subband_attentive_wavelet_packet_5d_decompose(...)`
- `subband_attentive_wavelet_packet_5d_reconstruct(...)`


## Long-memory temporal coupling

- `long_memory_temporal_head_coupling_operator(...)`

## Attentive multiscale gating

- `AttentiveMultiscaleHierarchicalGatedMemoryResult`
- `attentive_multiscale_hierarchical_hybrid_graph_temporal_gated_memory(...)`
- `attentive_multiscale_hierarchical_hybrid_graph_temporal_gated_stack(...)`

## Regularized and attentive packet trees

- `RegularizedAdaptiveWaveletPacket5DTree`
- `regularized_select_wavelet_family_per_axis_5d(...)`
- `regularized_adaptive_wavelet_packet_5d_decompose(...)`
- `regularized_adaptive_wavelet_packet_5d_reconstruct(...)`
- `SubbandAttentiveWaveletPacket5DTree`
- `subband_attentive_wavelet_packet_5d_decompose(...)`
- `subband_attentive_wavelet_packet_5d_reconstruct(...)`


## Temporal stability regularization

- `stability_regularized_temporal_head_coupling_operator(...)`

## Cross-branch tree attention

- `CrossBranchAttentiveWaveletPacket5DTree`
- `cross_branch_attentive_wavelet_packet_5d_decompose(...)`
- `cross_branch_attentive_wavelet_packet_5d_reconstruct(...)`

## Weighted multiobjective selection

- `WeightedMultiObjectiveWaveletPacket5DTree`
- `weighted_multiobjective_select_wavelet_family_per_axis_5d(...)`
- `weighted_multiobjective_wavelet_packet_5d_decompose(...)`
- `weighted_multiobjective_wavelet_packet_5d_reconstruct(...)`


## Joint temporal-spectral regularization

- `joint_temporal_spectral_regularized_coupling_operator(...)`

## Learnable multiobjective weights

- `LearnableWeightedMultiObjectiveWaveletPacket5DTree`
- `learnable_multiobjective_weight_search_5d(...)`
- `learnable_multiobjective_wavelet_packet_5d_decompose(...)`
- `learnable_multiobjective_wavelet_packet_5d_reconstruct(...)`

## Level-attentive packet trees

- `LevelAttentiveWaveletPacket5DTree`
- `level_attentive_wavelet_packet_5d_decompose(...)`
- `level_attentive_wavelet_packet_5d_reconstruct(...)`

## Forensic-ready toolkit

- `hash_bytes(payload)`
- `hash_file(path)`
- `create_evidence_manifest(source, case_id="", examiner="", notes="")`
- `get_forensic_profile(name)`
- `forensic_analyze_signal(source, profile="generic_signal", ...)`
- `write_forensic_bundle(result, output_dir)`

Produces:
- evidence manifest with MD5/SHA1/SHA256
- audit trail with per-step input/output SHA256
- reproducible JSON bundle (`manifest.json`, `audit_trail.json`, `analysis.json`, `forensic_report.json`)

## Formal forensic controls

- `create_chain_of_custody(...)`
- `append_chain_of_custody_event(...)`
- `create_timestamp_seal(source, authority="local_clock_untrusted", ...)`
- `sign_report(report_payload, signer="", signing_key=None)`

Notas:
- el sellado temporal actual es **local** y verificable por hash, no un TSA externo de confianza
- la firma actual puede ser `SHA256-DIGEST` o `HMAC-SHA256`
- los perfiles periciales por dominio incluyen audio, imagen scalarizada, red y multimodal

## Image decomposition

- `rgb_to_gray(image)`
- `mean_filter_2d(image, kernel_size=5)`
- `estimate_background(gray, kernel_size=15, smooth_kernel_size=5)`
- `extract_foreground(gray, background, mode="dark" | "bright")`
- `detect_reflections(gray, percentile=98.0)`
- `denoise_image(gray, kernel_size=3)`
- `estimate_noise(gray, denoised)`
- `extract_texture(gray, background)`
- `simple_edges(gray, kernel_size=3)`
- `wavelet_subbands_2d(gray, family="haar", level=1)`
- `decompose_image_layers(image, ...)`
- `reconstruct_from_layers(...)`

## Image layer export and forensic image pipeline

- `save_layer_image(layer, path)`
- `save_decomposition_layers(decomposition, output_dir, prefix="layer", image_format="png")`
- `estimate_illumination(gray, kernel_size=31)`
- `decompose_shadows_specular(gray, illumination, shadow_strength=0.08, specular_percentile=99.0)`
- `forensic_decompose_image(source, ..., output_dir=...)`

El pipeline forense de imagen genera:
- `manifest.json`
- `decomposition.json`
- `layer_hashes.json`
- `chain_of_custody.json`
- `forensic_image_report.json`
- carpeta `layers/` con las capas guardadas como imágenes
