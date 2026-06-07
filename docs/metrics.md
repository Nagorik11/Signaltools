# Módulo de Métricas y Evaluación de Calidad

El módulo `metrics` de `signaltools` proporciona un conjunto completo de métricas para evaluar la calidad de señales e imágenes procesadas.

## Funcionalidades Principales

### Métricas Individuales

- **MSE** (Mean Squared Error): Error cuadrático medio
- **RMSE** (Root Mean Squared Error): Raíz del error cuadrático medio
- **MAE** (Mean Absolute Error): Error absoluto medio
- **PSNR** (Peak Signal-to-Noise Ratio): Relación señal-ruido pico en dB
- **SSIM** (Structural Similarity Index): Índice de similitud estructural
- **NCC** (Normalized Cross-Correlation): Correlación cruzada normalizada
- **SNR** (Signal-to-Noise Ratio): Relación señal-ruido en dB
- **ISNR** (Improved SNR): Mejora en SNR después de denoising
- **VIF** (Visual Information Fidelity): Fidelidad de información visual

### Funciones Principales

#### `compute_all_metrics(original, processed)`
Calcula todas las métricas disponibles y devuelve un objeto `QualityMetrics`.

```python
from signaltools import compute_all_metrics
import numpy as np

original = np.random.rand(256, 256)
procesada = original + np.random.normal(0, 0.05, original.shape)

metrics = compute_all_metrics(original, procesada)
print(f"PSNR: {metrics.psnr:.2f} dB")
print(f"SSIM: {metrics.ssim:.4f}")
```

#### `print_metrics_report(metrics)`
Imprime un reporte formateado de todas las métricas.

```python
from signaltools import compute_all_metrics, print_metrics_report

metrics = compute_all_metrics(original, procesada)
print_metrics_report(metrics, title="Evaluación de Calidad")
```

#### `compare_signals(signals_dict)`
Compara múltiples señales contra una referencia.

```python
from signaltools import compare_signals

signals = {
    'reference': original,
    'denoised': denoisada,
    'enhanced': mejorada
}

results = compare_signals(signals)
for name, metrics in results.items():
    print(f"{name}: PSNR={metrics.psnr:.2f}, SSIM={metrics.ssim:.4f}")
```

## Ejemplo Completo

```python
import numpy as np
from signaltools import (
    mse, psnr, ssim, compute_all_metrics, 
    print_metrics_report, QualityMetrics
)

# Crear señal de prueba
original = np.sin(np.linspace(0, 4*np.pi, 1000))
ruido = np.random.normal(0, 0.1, original.shape)
procesada = original + ruido

# Calcular métricas individuales
print("MSE:", mse(original, procesada))
print("PSNR:", psnr(original, procesada), "dB")
print("SSIM:", ssim(original, procesada))

# Calcular todas las métricas
metrics = compute_all_metrics(original, procesada)
print_metrics_report(metrics)

# Acceder a métricas específicas
print(f"RMSE: {metrics.rmse:.4f}")
print(f"NCC: {metrics.ncc:.4f}")

# Convertir a diccionario
metrics_dict = metrics.to_dict()
```

## Integración con Pipelines

Las métricas se pueden usar fácilmente dentro de pipelines de procesamiento:

```python
from signaltools import ProcessingPipeline, compute_all_metrics

def evaluate_quality(step_result, context):
    original = context['original']
    processed = step_result.output
    context['metrics'] = compute_all_metrics(original, processed)
    return step_result.output

pipeline = ProcessingPipeline()
pipeline.add_step("denoise", tu_funcion_denoise)
pipeline.add_step("evaluate", evaluate_quality)

result = pipeline.execute(signal, context={'original': signal_original})
print(result.context['metrics'])
```

## Consideraciones

- **SSIM**: Para imágenes multicanal (RGB), calcula el promedio sobre los canales
- **PSNR**: Valores típicos: >30 dB es bueno, >40 dB es excelente
- **SSIM**: Rango [0, 1], donde 1 indica identidad perfecta
- **VIF**: Cálculo más costoso, opcional en `compute_all_metrics(compute_vif=True)`
- Todas las métricas soportan señales 1D e imágenes 2D/3D

## Referencias

- Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing.
- Hore, A., & Ziou, D. (2010). Image quality metrics: PSNR vs. SSIM.
