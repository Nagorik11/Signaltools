# =============================================================================
# SIGNALTOOLS - Metrics and Quality Assessment
# =============================================================================
"""
Módulo de métricas y evaluación de calidad para señales e imágenes.

Incluye métricas comunes como:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- SNR (Signal-to-Noise Ratio)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- NCC (Normalized Cross-Correlation)
- Y me más...

Autor: signaltools team
"""

import numpy as np
from typing import Union, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings


@dataclass
class QualityMetrics:
    """Resultado de múltiples métricas de calidad."""
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    ncc: Optional[float] = None
    snr: Optional[float] = None
    isnr: Optional[float] = None  # Improved SNR
    vif: Optional[float] = None   # Visual Information Fidelity
    
    def to_dict(self) -> Dict[str, float]:
        """Convertir a diccionario."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def __repr__(self):
        metrics = self.to_dict()
        if not metrics:
            return "QualityMetrics(no metrics computed)"
        items = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        return f"QualityMetrics({items})"


def _validate_inputs(
    original: np.ndarray, 
    processed: np.ndarray,
    check_shape: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Validar y preparar inputs para cálculo de métricas."""
    orig = np.asarray(original, dtype=np.float64)
    proc = np.asarray(processed, dtype=np.float64)
    
    if orig.shape != proc.shape:
        raise ValueError(
            f"Las formas no coinciden: original={orig.shape}, processed={proc.shape}"
        )
    
    if orig.size == 0:
        raise ValueError("Los arrays no pueden estar vacíos")
    
    return orig, proc


def mse(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calcular Mean Squared Error (MSE).
    
    Parámetros:
    -----------
    original : np.ndarray
        Señal o imagen original (referencia)
    processed : np.ndarray
        Señal o imagen procesada
    
    Retorna:
    --------
    float : Valor de MSE
    """
    orig, proc = _validate_inputs(original, processed)
    return np.mean((orig - proc) ** 2)


def rmse(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calcular Root Mean Squared Error (RMSE).
    
    Parámetros:
    -----------
    original : np.ndarray
        Señal o imagen original (referencia)
    processed : np.ndarray
        Señal o imagen procesada
    
    Retorna:
    --------
    float : Valor de RMSE
    """
    return np.sqrt(mse(original, processed))


def mae(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calcular Mean Absolute Error (MAE).
    
    Parámetros:
    -----------
    original : np.ndarray
        Señal o imagen original (referencia)
    processed : np.ndarray
        Señal o imagen procesada
    
    Retorna:
    --------
    float : Valor de MAE
    """
    orig, proc = _validate_inputs(original, processed)
    return np.mean(np.abs(orig - proc))


def psnr(
    original: np.ndarray, 
    processed: np.ndarray,
    data_range: Optional[float] = None
) -> float:
    """
    Calcular Peak Signal-to-Noise Ratio (PSNR) en dB.
    
    Parámetros:
    -----------
    original : np.ndarray
        Señal o imagen original (referencia)
    processed : np.ndarray
        Señal o imagen procesada
    data_range : float, opcional
        Rango dinámico de los datos. Si es None, se usa max - min del original.
    
    Retorna:
    --------
    float : Valor de PSNR en dB (mayor es mejor)
    """
    orig, proc = _validate_inputs(original, processed)
    
    if data_range is None:
        data_range = orig.max() - orig.min()
    
    if data_range == 0:
        return float('inf')  # Señal constante perfecta
    
    error_mse = mse(orig, proc)
    
    if error_mse == 0:
        return float('inf')  # Sin error
    
    return 10 * np.log10((data_range ** 2) / error_mse)


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Generar kernel gaussiano 1D."""
    x = np.arange(size) - (size - 1) / 2
    kernel = np.exp(-x**2 / (2 * sigma**2))
    return kernel / kernel.sum()


def _ssim_single_channel(
    orig: np.ndarray,
    proc: np.ndarray,
    data_range: float,
    K1: float = 0.01,
    K2: float = 0.03,
    L: float = 1.0,
    win_size: int = 7,
    sigma: float = 1.5
) -> float:
    """Calcular SSIM para un canal único."""
    C1 = (K1 * data_range * L) ** 2
    C2 = (K2 * data_range * L) ** 2
    
    # Kernel gaussiano
    kernel_1d = _gaussian_kernel(win_size, sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    
    # Preparar kernel para convolución según dimensionalidad
    if orig.ndim == 1:
        kernel = kernel_1d
    elif orig.ndim == 2:
        kernel = kernel_2d
    else:
        # Para 3D+, usar kernel separable
        kernel = kernel_2d
    
    # Función auxiliar para convolución local
    def local_mean(x):
        if x.ndim == 1:
            return np.convolve(x, kernel_1d, mode='valid')
        elif x.ndim == 2:
            from scipy.ndimage import convolve
            return convolve(x, kernel_2d, mode='constant')
        else:
            # Para dimensiones superiores, promediar en ventanas locales
            from scipy.ndimage import uniform_filter
            return uniform_filter(x, size=win_size)
    
    def local_sum_sq(x):
        return local_mean(x * x)
    
    def local_cross(x, y):
        return local_mean(x * y)
    
    mu_orig = local_mean(orig)
    mu_proc = local_mean(proc)
    
    mu_orig_sq = mu_orig * mu_orig
    mu_proc_sq = mu_proc * mu_proc
    mu_orig_proc = mu_orig * mu_proc
    
    sigma_orig_sq = local_sum_sq(orig) - mu_orig_sq
    sigma_proc_sq = local_sum_sq(proc) - mu_proc_sq
    sigma_orig_proc = local_cross(orig, proc) - mu_orig_proc
    
    # Asegurar que las varianzas sean no negativas
    sigma_orig_sq = np.maximum(sigma_orig_sq, 0)
    sigma_proc_sq = np.maximum(sigma_proc_sq, 0)
    
    numerador = (2 * mu_orig_proc + C1) * (2 * sigma_orig_proc + C2)
    denominador = (mu_orig_sq + mu_proc_sq + C1) * (sigma_orig_sq + sigma_proc_sq + C2)
    
    ssim_map = numerador / denominador
    return np.mean(ssim_map)


def ssim(
    original: np.ndarray,
    processed: np.ndarray,
    data_range: Optional[float] = None,
    multichannel: bool = True,
    win_size: int = 7,
    sigma: float = 1.5
) -> float:
    """
    Calcular Structural Similarity Index (SSIM).
    
    El SSIM mide la similitud estructural entre dos imágenes/señales,
    considerando luminancia, contraste y estructura.
    
    Parámetros:
    -----------
    original : np.ndarray
        Señal o imagen original (referencia)
    processed : np.ndarray
        Señal o imagen procesada
    data_range : float, opcional
        Rango dinámico de los datos
    multichannel : bool
        Si True, calcular SSIM promedio sobre canales (para imágenes RGB)
    win_size : int
        Tamaño de la ventana para cálculo local
    sigma : float
        Desviación estándar del kernel gaussiano
    
    Retorna:
    --------
    float : Valor de SSIM entre 0 y 1 (más cercano a 1 es mejor)
    """
    orig, proc = _validate_inputs(original, processed)
    
    if data_range is None:
        data_range = orig.max() - orig.min()
    
    if data_range == 0:
        return 1.0 if np.allclose(orig, proc) else 0.0
    
    # Normalizar a [0, 1]
    orig_norm = (orig - orig.min()) / (data_range + 1e-10)
    proc_norm = (proc - proc.min()) / (data_range + 1e-10)
    
    # Manejar múltiples canales
    if multichannel and orig.ndim >= 3 and orig.shape[-1] <= 4:
        # Asumir último eje es canal
        ssim_values = []
        for c in range(orig.shape[-1]):
            ssim_val = _ssim_single_channel(
                orig_norm[..., c], proc_norm[..., c],
                data_range=1.0, win_size=win_size, sigma=sigma
            )
            ssim_values.append(ssim_val)
        return float(np.mean(ssim_values))
    else:
        return _ssim_single_channel(
            orig_norm, proc_norm,
            data_range=1.0, win_size=win_size, sigma=sigma
        )


def ncc(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calcular Normalized Cross-Correlation (NCC).
    
    Parámetros:
    -----------
    original : np.ndarray
        Señal o imagen original (referencia)
    processed : np.ndarray
        Señal o imagen procesada
    
    Retorna:
    --------
    float : Valor de NCC entre -1 y 1 (más cercano a 1 es mejor)
    """
    orig, proc = _validate_inputs(original, processed)
    
    orig_centered = orig - np.mean(orig)
    proc_centered = proc - np.mean(proc)
    
    numerator = np.sum(orig_centered * proc_centered)
    denominator = np.sqrt(
        np.sum(orig_centered ** 2) * np.sum(proc_centered ** 2)
    )
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def snr(original: np.ndarray, noise: Optional[np.ndarray] = None) -> float:
    """
    Calcular Signal-to-Noise Ratio (SNR) en dB.
    
    Parámetros:
    -----------
    original : np.ndarray
        Señal original (o señal limpia si se provee noise)
    noise : np.ndarray, opcional
        Señal de ruido. Si es None, se asume que original contiene ruido
        y se estima el ruido como la diferencia con una versión suavizada.
    
    Retorna:
    --------
    float : Valor de SNR en dB (mayor es mejor)
    """
    orig = np.asarray(original, dtype=np.float64)
    
    if noise is not None:
        noise = np.asarray(noise, dtype=np.float64)
        if orig.shape != noise.shape:
            raise ValueError("original y noise deben tener la misma forma")
        signal_power = np.mean(orig ** 2)
        noise_power = np.mean(noise ** 2)
    else:
        # Estimar ruido usando filtro mediano como aproximación de señal limpia
        from scipy.ndimage import median_filter
        signal_estimate = median_filter(orig, size=3)
        noise_estimate = orig - signal_estimate
        
        signal_power = np.mean(signal_estimate ** 2)
        noise_power = np.mean(noise_estimate ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)


def isnr(
    original: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray
) -> float:
    """
    Calcular Improved SNR (ISNR) en dB.
    
    Mide la mejora en SNR después de aplicar un algoritmo de denoising.
    
    Parámetros:
    -----------
    original : np.ndarray
        Señal original limpia (ground truth)
    noisy : np.ndarray
        Señal con ruido
    denoised : np.ndarray
        Señal después de aplicar denoising
    
    Retorna:
    --------
    float : Valor de ISNR en dB (positivo indica mejora)
    """
    orig, noisy = _validate_inputs(original, noisy)
    _, denoised = _validate_inputs(original, denoised)
    
    # SNR antes del denoising
    noise_before = noisy - orig
    snr_before = snr(orig, noise_before)
    
    # SNR después del denoising
    noise_after = denoised - orig
    snr_after = snr(orig, noise_after)
    
    return snr_after - snr_before


def vif(
    original: np.ndarray,
    processed: np.ndarray,
    sigma_n: float = 0.0001
) -> float:
    """
    Calcular Visual Information Fidelity (VIF) simplificado.
    
    Nota: Esta es una implementación simplificada. La versión completa
    requiere modelos del sistema visual humano más complejos.
    
    Parámetros:
    -----------
    original : np.ndarray
        Señal o imagen original (referencia)
    processed : np.ndarray
        Señal o imagen procesada
    sigma_n : float
        Varianza del ruido del canal
    
    Retorna:
    --------
    float : Valor de VIF (mayor es mejor, típicamente > 0)
    """
    orig, proc = _validate_inputs(original, processed)
    
    # Normalizar
    orig_norm = (orig - orig.mean()) / (orig.std() + 1e-10)
    proc_norm = (proc - proc.mean()) / (proc.std() + 1e-10)
    
    # Calcular varianza local
    from scipy.ndimage import uniform_filter
    
    def local_variance(x):
        mean_x = uniform_filter(x, size=3)
        mean_x_sq = uniform_filter(x ** 2, size=3)
        return mean_x_sq - mean_x ** 2
    
    var_orig = local_variance(orig_norm)
    var_proc = local_variance(proc_norm)
    
    # Covarianza local
    mean_orig_proc = uniform_filter(orig_norm * proc_norm, size=3)
    mean_orig = uniform_filter(orig_norm, size=3)
    mean_proc = uniform_filter(proc_norm, size=3)
    cov_orig_proc = mean_orig_proc - mean_orig * mean_proc
    
    # Calcular VIF
    epsilon = 1e-10
    vif_num = np.log2(1 + (var_orig * cov_orig_proc ** 2) / (var_orig * sigma_n ** 2 + epsilon))
    vif_den = np.log2(1 + var_orig / sigma_n ** 2)
    
    vif_map = vif_num / (vif_den + epsilon)
    
    return float(np.nanmean(vif_map))


def compute_all_metrics(
    original: np.ndarray,
    processed: np.ndarray,
    data_range: Optional[float] = None,
    compute_vif: bool = False
) -> QualityMetrics:
    """
    Calcular todas las métricas de calidad disponibles.
    
    Parámetros:
    -----------
    original : np.ndarray
        Señal o imagen original (referencia)
    processed : np.ndarray
        Señal o imagen procesada
    data_range : float, opcional
        Rango dinámico de los datos
    compute_vif : bool
        Si True, calcular también VIF (más costoso computacionalmente)
    
    Retorna:
    --------
    QualityMetrics : Objeto con todas las métricas calculadas
    """
    orig, proc = _validate_inputs(original, processed)
    
    if data_range is None:
        data_range = orig.max() - orig.min()
    
    metrics = QualityMetrics()
    
    # Métricas básicas de error
    metrics.mse = mse(orig, proc)
    metrics.rmse = np.sqrt(metrics.mse)
    metrics.mae = mae(orig, proc)
    
    # PSNR
    metrics.psnr = psnr(orig, proc, data_range=data_range)
    
    # SSIM
    metrics.ssim = ssim(orig, proc, data_range=data_range)
    
    # NCC
    metrics.ncc = ncc(orig, proc)
    
    # VIF (opcional, más costoso)
    if compute_vif:
        try:
            metrics.vif = vif(orig, proc)
        except Exception as e:
            warnings.warn(f"No se pudo calcular VIF: {e}")
    
    return metrics


def compare_signals(
    signals: Dict[str, np.ndarray],
    reference_name: str = 'reference'
) -> Dict[str, QualityMetrics]:
    """
    Comparar múltiples señales contra una referencia.
    
    Parámetros:
    -----------
    signals : dict
        Diccionario con nombres de señales y sus arrays
    reference_name : str
        Nombre de la señal de referencia en el diccionario
    
    Retorna:
    --------
    dict : Diccionario con nombre de señal -> QualityMetrics
    """
    if reference_name not in signals:
        raise ValueError(f"La señal de referencia '{reference_name}' no está en signals")
    
    reference = signals[reference_name]
    results = {}
    
    for name, signal in signals.items():
        if name == reference_name:
            continue
        try:
            results[name] = compute_all_metrics(reference, signal)
        except Exception as e:
            warnings.warn(f"No se pudo comparar {name}: {e}")
    
    return results


def print_metrics_report(metrics: QualityMetrics, title: str = "Quality Metrics Report") -> None:
    """
    Imprimir reporte formateado de métricas.
    
    Parámetros:
    -----------
    metrics : QualityMetrics
        Objeto con las métricas calculadas
    title : str
        Título del reporte
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    metric_descriptions = {
        'mse': ('MSE', 'Mean Squared Error', 'menor es mejor'),
        'rmse': ('RMSE', 'Root Mean Squared Error', 'menor es mejor'),
        'mae': ('MAE', 'Mean Absolute Error', 'menor es mejor'),
        'psnr': ('PSNR', 'Peak Signal-to-Noise Ratio (dB)', 'mayor es mejor'),
        'ssim': ('SSIM', 'Structural Similarity Index', 'más cercano a 1 es mejor'),
        'ncc': ('NCC', 'Normalized Cross-Correlation', 'más cercano a 1 es mejor'),
        'snr': ('SNR', 'Signal-to-Noise Ratio (dB)', 'mayor es mejor'),
        'isnr': ('ISNR', 'Improved SNR (dB)', 'positivo indica mejora'),
        'vif': ('VIF', 'Visual Information Fidelity', 'mayor es mejor'),
    }
    
    for key, (abbr, desc, interpretation) in metric_descriptions.items():
        value = getattr(metrics, key)
        if value is not None:
            if np.isinf(value):
                print(f"{abbr:8s}: {'∞':>12s} ({interpretation})")
                print(f"         {desc}")
            else:
                print(f"{abbr:8s}: {value:>12.4f} ({interpretation})")
                print(f"         {desc}")
    
    print(f"{'='*60}\n")


__all__ = [
    # Clases
    'QualityMetrics',
    
    # Métricas individuales
    'mse',
    'rmse',
    'mae',
    'psnr',
    'ssim',
    'ncc',
    'snr',
    'isnr',
    'vif',
    
    # Funciones utilitarias
    'compute_all_metrics',
    'compare_signals',
    'print_metrics_report',
]
