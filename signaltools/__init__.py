# =============================================================================
# SIGNALTOOLS - Public API
# =============================================================================
# Core signal processing and I/O
from .io import read_signal_file, guess_numeric_views, write_wav, read_wav, read_audio_file, Ingestor, SignalBuffer
from .framing import FrameConfig, frame_signal, normalize_signal, detrend_mean, standardize_signal
from .features import frame_feature_vector, first_derivative, second_derivative
from .exceptions import SignalToolsError, SignalValidationError
from .logging_utils import configure_logging, get_logger

# Spectral analysis and detection
from .spectral import (
    dft,
    dominant_bins,
    spectral_energy,
    spectral_flatness,
    frequency_axis,
    power_spectrum,
    spectral_centroid,
    spectral_bandwidth,
    spectral_rolloff,
    band_energy,
    stft,
    spectrogram_matrix,
    autocorrelation,
    estimate_pitch,
)
from .detect import threshold_events, adaptive_threshold, adaptive_events, local_peaks, anomaly_score, onset_strength

# Basic filters and filter design
from .filters import moving_average, median_filter, remove_dc, normalize_peak, fft_bandpass
from .filter_design import (
    FIRCoefficients,
    IIRCoefficients,
    AdaptiveFilterResult,
    apply_fir,
    apply_iir,
    fir_lowpass,
    fir_highpass,
    fir_bandpass,
    fir_bandstop,
    fractional_delay_fir,
    differentiator_fir,
    iir_integrator_leaky,
    iir_lowpass_single_pole,
    iir_highpass_single_pole,
    biquad_lowpass,
    biquad_highpass,
    biquad_bandpass,
    biquad_notch,
    biquad_allpass,
    comb_filter_feedforward,
    comb_filter_feedback,
    savitzky_golay_coefficients,
    savitzky_golay_filter,
    hilbert_transform_fft,
    analytic_signal,
    envelope,
    lms_adaptive_filter,
)

# Morphological operations (1D)
from .morphology import (
    advanced_median_filter,
    rank_filter,
    dilation_1d,
    erosion_1d,
    opening_1d,
    closing_1d,
    morphological_gradient_1d,
)

# State estimation filters
from .state_filters import (
    KalmanFilterResult,
    WienerFilterResult,
    kalman_filter_1d,
    wiener_filter_1d,
)
from .advanced_state_filters import (
    AdaptiveWienerResult,
    NonlinearFilterResult,
    ParticleFilterResult,
    SmootherResult,
    adaptive_wiener_filter_1d,
    extended_kalman_filter,
    unscented_kalman_filter,
    particle_filter_1d,
    backward_exponential_smoother,
    rts_smoother,
    particle_filter_nonlinear,
    particle_filter_multivariate,
)

# Multirate and filter banks
from .multirate import (
    polyphase_decompose,
    decimate,
    interpolate,
    two_band_analysis_bank,
)
from .filter_banks import (
    FilterBankResult,
    haar_analysis_bank,
    haar_synthesis_bank,
    uniform_filter_bank,
    reconstruct_uniform_filter_bank,
)

# Modulation and fingerprinting
from .modulate import amplitude_modulation, frequency_modulation
from .fingerprint import SignalFingerprint, fingerprint_engine, cosine_similarity, euclidean_distance, compare_fingerprints

# Bridge and bitlayer analysis
from .bridge import (
    signal_signature,
    SignalSignature,
    signature_to_glyph_vector,
    LayeredSignalAnalysis,
    analyze_signal_layered,
    reconstruct_signal_from_signature,
)
from .bitlayer import analyze_bitlayer, build_bit_signature, BitSignature

# Pipeline and manager
from .pipeline import AdvancedSignalAnalysis, analyze_signal_advanced

# Graph signal processing
from .graph_filters import (
    graph_laplacian,
    graph_fourier_basis,
    graph_filter_signal,
    graph_polynomial_filter,
)
from .graph_wavelets import (
    chebyshev_graph_filter,
    graph_wavelet_kernel,
    graph_wavelet_transform,
)
from .graph_positional import (
    laplacian_positional_encoding,
    random_walk_positional_encoding,
    augment_with_graph_positional_encoding,
)

# Deep graph neural networks
from .graph_deep_filters import (
    GNNStackResult,
    MultiHeadAttentionResult,
    MultiHeadNodeAttentionResult,
    DeepGNNResult,
    GraphTransformerResult,
    EdgeConditionedConvResult,
    MessagePassingResult,
    QKVAttentionResult,
    ChannelMixResult,
    HybridTemporalAttentionResult,
    RecurrentHybridAttentionResult,
    HybridGatedMemoryResult,
    BidirectionalGatedMemoryResult,
    HierarchicalGatedMemoryResult,
    MultiscaleHierarchicalGatedMemoryResult,
    AttentiveMultiscaleHierarchicalGatedMemoryResult,
    graph_block_normalize,
    channel_mix,
    graph_block_normalize_multichannel,
    edge_aware_message_passing,
    edge_conditioned_convolution,
    edge_conditioned_conv_stack,
    edge_feature_message_passing_stack,
    graph_pool,
    graph_attention_matrix,
    graph_attention_filter,
    multihead_graph_attention,
    multihead_graph_attention_multichannel,
    qkv_graph_attention,
    masked_qkv_graph_attention,
    structured_edge_embedding_attention,
    spectral_gnn_filter,
    spectral_gnn_filter_multichannel,
    graph_scattering_transform,
    stacked_gnn,
    deep_gnn_stack,
    graph_transformer_layer,
    graph_transformer_stack,
    graph_transformer_qkv_layer,
    graph_transformer_qkv_stack,
    graph_transformer_masked_qkv_layer,
    graph_transformer_masked_qkv_stack,
    graph_transformer_enhanced_layer,
    graph_transformer_enhanced_stack,
    graph_transformer_edge_embedding_layer,
    graph_transformer_edge_embedding_stack,
    hybrid_node_edge_temporal_attention,
    hybrid_graph_temporal_transformer_layer,
    hybrid_graph_temporal_transformer_stack,
    recurrent_hybrid_node_edge_temporal_attention,
    recurrent_hybrid_graph_temporal_transformer_stack,
    hybrid_graph_temporal_gated_memory,
    hybrid_graph_temporal_gated_stack,
    bidirectional_hybrid_graph_temporal_gated_memory,
    bidirectional_hybrid_graph_temporal_gated_stack,
    hierarchical_hybrid_graph_temporal_gated_memory,
    hierarchical_hybrid_graph_temporal_gated_stack,
    multiscale_hierarchical_hybrid_graph_temporal_gated_memory,
    multiscale_hierarchical_hybrid_graph_temporal_gated_stack,
    attentive_multiscale_hierarchical_hybrid_graph_temporal_gated_memory,
    attentive_multiscale_hierarchical_hybrid_graph_temporal_gated_stack,
)

# Wavelet transforms (1D, 2D, 3D, 4D, 5D)
from .wavelet_packet import (
    WaveletPacketTree,
    available_wavelet_families,
    wavelet_family_kind,
    wavelet_filters,
    wavelet_packet_decompose,
    wavelet_packet_reconstruct,
)
from .wavelet_packet_2d import (
    WaveletPacket2DTree,
    wavelet_packet_2d_decompose,
    wavelet_packet_2d_reconstruct,
)
from .wavelet_packet_3d import (
    WaveletPacket3DTree,
    AnisotropicWaveletPacket3DTree,
    wavelet_packet_3d_decompose,
    wavelet_packet_3d_reconstruct,
    anisotropic_wavelet_packet_3d_decompose,
    anisotropic_wavelet_packet_3d_reconstruct,
)
from .wavelet_packet_4d import (
    WaveletPacket4DTree,
    AnisotropicWaveletPacket4DTree,
    wavelet_packet_4d_decompose,
    wavelet_packet_4d_reconstruct,
    anisotropic_wavelet_packet_4d_decompose,
    anisotropic_wavelet_packet_4d_reconstruct,
)
from .wavelet_packet_5d import (
    WaveletPacket5DTree,
    AnisotropicWaveletPacket5DTree,
    AdaptiveWaveletPacket5DTree,
    BlockAdaptiveWaveletPacket5DTree,
    SpatiallyAdaptiveWaveletPacket5DTree,
    SubbandAdaptiveWaveletPacket5DTree,
    RegularizedAdaptiveWaveletPacket5DTree,
    SubbandAttentiveWaveletPacket5DTree,
    CrossBranchAttentiveWaveletPacket5DTree,
    WeightedMultiObjectiveWaveletPacket5DTree,
    LearnableWeightedMultiObjectiveWaveletPacket5DTree,
    LevelAttentiveWaveletPacket5DTree,
    wavelet_packet_5d_decompose,
    wavelet_packet_5d_reconstruct,
    anisotropic_wavelet_packet_5d_decompose,
    anisotropic_wavelet_packet_5d_reconstruct,
    select_wavelet_family_per_axis_5d,
    adaptive_wavelet_packet_5d_decompose,
    adaptive_wavelet_packet_5d_reconstruct,
    select_wavelet_family_per_axis_5d_local_blocks,
    adaptive_blockwise_wavelet_packet_5d_decompose,
    adaptive_blockwise_wavelet_packet_5d_reconstruct,
    spatially_variable_wavelet_packet_5d_decompose,
    spatially_variable_wavelet_packet_5d_reconstruct,
    subband_adaptive_wavelet_packet_5d_decompose,
    subband_adaptive_wavelet_packet_5d_reconstruct,
    regularized_select_wavelet_family_per_axis_5d,
    regularized_adaptive_wavelet_packet_5d_decompose,
    regularized_adaptive_wavelet_packet_5d_reconstruct,
    subband_attentive_wavelet_packet_5d_decompose,
    subband_attentive_wavelet_packet_5d_reconstruct,
    cross_branch_attentive_wavelet_packet_5d_decompose,
    cross_branch_attentive_wavelet_packet_5d_reconstruct,
    weighted_multiobjective_select_wavelet_family_per_axis_5d,
    weighted_multiobjective_wavelet_packet_5d_decompose,
    weighted_multiobjective_wavelet_packet_5d_reconstruct,
    learnable_multiobjective_weight_search_5d,
    learnable_multiobjective_wavelet_packet_5d_decompose,
    learnable_multiobjective_wavelet_packet_5d_reconstruct,
    level_attentive_wavelet_packet_5d_decompose,
    level_attentive_wavelet_packet_5d_reconstruct,
)

# Image processing and forensics
from .image_decomposition import (
    ImageLayerDecomposition,
    SavedLayerImages,
    WaveletSubbands2D,
    build_layer_alpha_masks,
    decompose_image_layers,
    decompose_shadows_specular,
    denoise_image,
    detect_reflections,
    estimate_background,
    estimate_illumination,
    estimate_noise,
    extract_foreground,
    extract_texture,
    mean_filter_2d,
    reconstruct_from_layers,
    rgb_to_gray,
    save_alpha_masks,
    save_decomposition_layers,
    save_layer_image,
    segment_ink_strokes,
    simple_edges,
    wavelet_subbands_2d,
)
from .image_visualization import (
    ComparisonMosaicResult,
    export_comparison_mosaic,
)
from .image_forensics import (
    ForensicImageAnalysisResult,
    ForensicImageBundlePaths,
    forensic_decompose_image,
)
from .image_morphology import (
    dilation_2d,
    erosion_2d,
    opening_2d,
    closing_2d,
    median_filter_2d,
    morphological_gradient_2d,
    dilation_3d,
    erosion_3d,
    opening_3d,
    closing_3d,
    median_filter_3d,
    morphological_gradient_3d,
    dilation_3d_kernel,
    erosion_3d_kernel,
    opening_3d_kernel,
    closing_3d_kernel,
)

# Complex-valued deep learning operators
from .complex_multichannel import (
    ComplexChannelResult,
    analytic_signal_multichannel,
    complex_channel_mix,
)
from .complex_spectral import (
    ComplexSpectralResult,
    complex_dft_multichannel,
    complex_spectral_mask,
    complex_spectral_shift,
)
from .complex_frame import (
    ComplexFrameResult,
    complex_stft_multichannel,
    complex_frame_operator,
)
from .complex_learnable_tf import (
    ComplexLearnableTFResult,
    complex_learnable_tf_operator,
    complex_learnable_tf_stack,
)
from .complex_multihead_tf import (
    ComplexMultiHeadTFResult,
    multihead_band_complex_tf_operator,
    multihead_band_complex_tf_stack,
)
from .complex_attention_tf import (
    ComplexCoupledAttentionTFResult,
    complex_multiband_head_coupling_operator,
    complex_multiband_head_coupling_stack,
    temporal_complex_head_coupling_operator,
    content_conditioned_temporal_head_coupling_operator,
    mode_conditioned_temporal_head_coupling_operator,
    long_memory_temporal_head_coupling_operator,
    stability_regularized_temporal_head_coupling_operator,
    joint_temporal_spectral_regularized_coupling_operator,
)

# Forensics and audit
from .forensics import (
    AuditStep,
    ChainOfCustody,
    ChainOfCustodyEvent,
    EvidenceHashes,
    EvidenceManifest,
    ForensicAnalysisResult,
    ForensicBundlePaths,
    ForensicProfile,
    FORENSIC_PROFILES,
    ReportSignature,
    TimestampSeal,
    append_chain_of_custody_event,
    create_chain_of_custody,
    create_evidence_manifest,
    create_timestamp_seal,
    forensic_analyze_signal,
    get_forensic_profile,
    hash_bytes,
    hash_file,
    sign_report,
    write_forensic_bundle,
)

__all__ = [
    # Core I/O and framing
    "read_signal_file", "guess_numeric_views", "write_wav", "read_wav", "read_audio_file", "Ingestor", "SignalBuffer",
    "FrameConfig", "frame_signal", "normalize_signal", "detrend_mean", "standardize_signal",
    "frame_feature_vector", "first_derivative", "second_derivative",
    
    # Spectral analysis
    "dft", "dominant_bins", "spectral_energy", "spectral_flatness", "frequency_axis", "power_spectrum",
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "band_energy", "stft", "spectrogram_matrix",
    "autocorrelation", "estimate_pitch",
    
    # Detection
    "threshold_events", "adaptive_threshold", "adaptive_events", "local_peaks", "anomaly_score", "onset_strength",
    
    # Filters and filter design
    "moving_average", "median_filter", "remove_dc", "normalize_peak", "fft_bandpass",
    "FIRCoefficients", "IIRCoefficients", "AdaptiveFilterResult",
    "apply_fir", "apply_iir",
    "fir_lowpass", "fir_highpass", "fir_bandpass", "fir_bandstop",
    "fractional_delay_fir", "differentiator_fir",
    "iir_integrator_leaky", "iir_lowpass_single_pole", "iir_highpass_single_pole",
    "biquad_lowpass", "biquad_highpass", "biquad_bandpass", "biquad_notch", "biquad_allpass",
    "comb_filter_feedforward", "comb_filter_feedback",
    "savitzky_golay_coefficients", "savitzky_golay_filter",
    "hilbert_transform_fft", "analytic_signal", "envelope", "lms_adaptive_filter",
    
    # Morphology (1D)
    "advanced_median_filter", "rank_filter", "dilation_1d", "erosion_1d", "opening_1d", "closing_1d", "morphological_gradient_1d",
    
    # State estimation
    "KalmanFilterResult", "WienerFilterResult", "kalman_filter_1d", "wiener_filter_1d",
    "AdaptiveWienerResult", "NonlinearFilterResult", "ParticleFilterResult", "SmootherResult",
    "adaptive_wiener_filter_1d", "extended_kalman_filter", "unscented_kalman_filter", "particle_filter_1d",
    "backward_exponential_smoother", "rts_smoother", "particle_filter_nonlinear", "particle_filter_multivariate",
    
    # Multirate and filter banks
    "polyphase_decompose", "decimate", "interpolate", "two_band_analysis_bank",
    "FilterBankResult", "haar_analysis_bank", "haar_synthesis_bank", "uniform_filter_bank", "reconstruct_uniform_filter_bank",
    
    # Modulation and fingerprinting
    "amplitude_modulation", "frequency_modulation",
    "SignalFingerprint", "fingerprint_engine", "cosine_similarity", "euclidean_distance", "compare_fingerprints",
    
    # Bridge and bitlayer
    "signal_signature", "SignalSignature", "signature_to_glyph_vector", "LayeredSignalAnalysis",
    "analyze_signal_layered", "reconstruct_signal_from_signature",
    "analyze_bitlayer", "build_bit_signature", "BitSignature",
    
    # Pipeline
    "AdvancedSignalAnalysis", "analyze_signal_advanced",
    
    # Graph signal processing
    "graph_laplacian", "graph_fourier_basis", "graph_filter_signal", "graph_polynomial_filter",
    "chebyshev_graph_filter", "graph_wavelet_kernel", "graph_wavelet_transform",
    "laplacian_positional_encoding", "random_walk_positional_encoding", "augment_with_graph_positional_encoding",
    
    # Deep graph neural networks
    "GNNStackResult", "MultiHeadAttentionResult", "MultiHeadNodeAttentionResult", "DeepGNNResult", 
    "GraphTransformerResult", "EdgeConditionedConvResult", "MessagePassingResult", "QKVAttentionResult", 
    "ChannelMixResult", "HybridTemporalAttentionResult", "RecurrentHybridAttentionResult",
    "HybridGatedMemoryResult", "BidirectionalGatedMemoryResult", "HierarchicalGatedMemoryResult",
    "MultiscaleHierarchicalGatedMemoryResult", "AttentiveMultiscaleHierarchicalGatedMemoryResult",
    "graph_block_normalize", "channel_mix", "graph_block_normalize_multichannel", 
    "edge_aware_message_passing", "edge_conditioned_convolution", "edge_conditioned_conv_stack", 
    "edge_feature_message_passing_stack", "graph_pool", "graph_attention_matrix", "graph_attention_filter", 
    "multihead_graph_attention", "multihead_graph_attention_multichannel", "qkv_graph_attention", 
    "masked_qkv_graph_attention", "structured_edge_embedding_attention", "spectral_gnn_filter", 
    "spectral_gnn_filter_multichannel", "graph_scattering_transform", "stacked_gnn", "deep_gnn_stack", 
    "graph_transformer_layer", "graph_transformer_stack", "graph_transformer_qkv_layer", 
    "graph_transformer_qkv_stack", "graph_transformer_masked_qkv_layer", "graph_transformer_masked_qkv_stack", 
    "graph_transformer_enhanced_layer", "graph_transformer_enhanced_stack", "graph_transformer_edge_embedding_layer", 
    "graph_transformer_edge_embedding_stack", "hybrid_node_edge_temporal_attention", 
    "hybrid_graph_temporal_transformer_layer", "hybrid_graph_temporal_transformer_stack",
    "recurrent_hybrid_node_edge_temporal_attention", "recurrent_hybrid_graph_temporal_transformer_stack",
    "hybrid_graph_temporal_gated_memory", "hybrid_graph_temporal_gated_stack",
    "bidirectional_hybrid_graph_temporal_gated_memory", "bidirectional_hybrid_graph_temporal_gated_stack",
    "hierarchical_hybrid_graph_temporal_gated_memory", "hierarchical_hybrid_graph_temporal_gated_stack",
    "multiscale_hierarchical_hybrid_graph_temporal_gated_memory", "multiscale_hierarchical_hybrid_graph_temporal_gated_stack",
    "attentive_multiscale_hierarchical_hybrid_graph_temporal_gated_memory", "attentive_multiscale_hierarchical_hybrid_graph_temporal_gated_stack",
    
    # Wavelet transforms (1D-5D)
    "WaveletPacketTree", "available_wavelet_families", "wavelet_family_kind", "wavelet_filters", 
    "wavelet_packet_decompose", "wavelet_packet_reconstruct",
    "WaveletPacket2DTree", "wavelet_packet_2d_decompose", "wavelet_packet_2d_reconstruct",
    "WaveletPacket3DTree", "AnisotropicWaveletPacket3DTree", "wavelet_packet_3d_decompose", 
    "wavelet_packet_3d_reconstruct", "anisotropic_wavelet_packet_3d_decompose", "anisotropic_wavelet_packet_3d_reconstruct",
    "WaveletPacket4DTree", "AnisotropicWaveletPacket4DTree", "wavelet_packet_4d_decompose", 
    "wavelet_packet_4d_reconstruct", "anisotropic_wavelet_packet_4d_decompose", "anisotropic_wavelet_packet_4d_reconstruct",
    "WaveletPacket5DTree", "AnisotropicWaveletPacket5DTree", "AdaptiveWaveletPacket5DTree",
    "BlockAdaptiveWaveletPacket5DTree", "SpatiallyAdaptiveWaveletPacket5DTree", "SubbandAdaptiveWaveletPacket5DTree",
    "RegularizedAdaptiveWaveletPacket5DTree", "SubbandAttentiveWaveletPacket5DTree", "CrossBranchAttentiveWaveletPacket5DTree",
    "WeightedMultiObjectiveWaveletPacket5DTree", "LearnableWeightedMultiObjectiveWaveletPacket5DTree",
    "LevelAttentiveWaveletPacket5DTree",
    "wavelet_packet_5d_decompose", "wavelet_packet_5d_reconstruct",
    "anisotropic_wavelet_packet_5d_decompose", "anisotropic_wavelet_packet_5d_reconstruct",
    "select_wavelet_family_per_axis_5d", "adaptive_wavelet_packet_5d_decompose", "adaptive_wavelet_packet_5d_reconstruct",
    "select_wavelet_family_per_axis_5d_local_blocks", "adaptive_blockwise_wavelet_packet_5d_decompose",
    "adaptive_blockwise_wavelet_packet_5d_reconstruct", "spatially_variable_wavelet_packet_5d_decompose",
    "spatially_variable_wavelet_packet_5d_reconstruct", "subband_adaptive_wavelet_packet_5d_decompose",
    "subband_adaptive_wavelet_packet_5d_reconstruct", "regularized_select_wavelet_family_per_axis_5d",
    "regularized_adaptive_wavelet_packet_5d_decompose", "regularized_adaptive_wavelet_packet_5d_reconstruct",
    "subband_attentive_wavelet_packet_5d_decompose", "subband_attentive_wavelet_packet_5d_reconstruct",
    "cross_branch_attentive_wavelet_packet_5d_decompose", "cross_branch_attentive_wavelet_packet_5d_reconstruct",
    "weighted_multiobjective_select_wavelet_family_per_axis_5d", "weighted_multiobjective_wavelet_packet_5d_decompose",
    "weighted_multiobjective_wavelet_packet_5d_reconstruct", "learnable_multiobjective_weight_search_5d",
    "learnable_multiobjective_wavelet_packet_5d_decompose", "learnable_multiobjective_wavelet_packet_5d_reconstruct",
    "level_attentive_wavelet_packet_5d_decompose", "level_attentive_wavelet_packet_5d_reconstruct",
    
    # Image processing
    "ImageLayerDecomposition", "SavedLayerImages", "WaveletSubbands2D",
    "build_layer_alpha_masks", "decompose_image_layers", "decompose_shadows_specular",
    "denoise_image", "detect_reflections", "estimate_background", "estimate_illumination",
    "estimate_noise", "extract_foreground", "extract_texture", "mean_filter_2d",
    "reconstruct_from_layers", "rgb_to_gray", "save_alpha_masks", "save_decomposition_layers",
    "save_layer_image", "segment_ink_strokes", "simple_edges", "wavelet_subbands_2d",
    
    # Image visualization and forensics
    "ComparisonMosaicResult", "export_comparison_mosaic",
    "ForensicImageAnalysisResult", "ForensicImageBundlePaths", "forensic_decompose_image",
    
    # Image morphology (2D/3D)
    "dilation_2d", "erosion_2d", "opening_2d", "closing_2d", "median_filter_2d", "morphological_gradient_2d",
    "dilation_3d", "erosion_3d", "opening_3d", "closing_3d", "median_filter_3d", "morphological_gradient_3d",
    "dilation_3d_kernel", "erosion_3d_kernel", "opening_3d_kernel", "closing_3d_kernel",
    
    # Complex-valued deep learning
    "ComplexChannelResult", "analytic_signal_multichannel", "complex_channel_mix",
    "ComplexSpectralResult", "complex_dft_multichannel", "complex_spectral_mask", "complex_spectral_shift",
    "ComplexFrameResult", "complex_stft_multichannel", "complex_frame_operator",
    "ComplexLearnableTFResult", "complex_learnable_tf_operator", "complex_learnable_tf_stack",
    "ComplexMultiHeadTFResult", "multihead_band_complex_tf_operator", "multihead_band_complex_tf_stack",
    "ComplexCoupledAttentionTFResult", "complex_multiband_head_coupling_operator",
    "complex_multiband_head_coupling_stack", "temporal_complex_head_coupling_operator",
    "content_conditioned_temporal_head_coupling_operator", "mode_conditioned_temporal_head_coupling_operator",
    "long_memory_temporal_head_coupling_operator", "stability_regularized_temporal_head_coupling_operator",
    "joint_temporal_spectral_regularized_coupling_operator",
    
    # Forensics and audit
    "AuditStep", "ChainOfCustody", "ChainOfCustodyEvent", "EvidenceHashes", "EvidenceManifest",
    "ForensicAnalysisResult", "ForensicBundlePaths", "ForensicProfile", "FORENSIC_PROFILES",
    "ReportSignature", "TimestampSeal",
    "append_chain_of_custody_event", "create_chain_of_custody", "create_evidence_manifest",
    "create_timestamp_seal", "forensic_analyze_signal", "get_forensic_profile", "hash_bytes",
    "hash_file", "sign_report", "write_forensic_bundle",
    
    # Utilities
    "configure_logging", "get_logger",
    "SignalToolsError", "SignalValidationError",
]
