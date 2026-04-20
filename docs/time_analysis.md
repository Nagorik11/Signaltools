# Módulo Time Analysis (Análisis Temporal)

Este conjunto de módulos permite extraer información directamente de la forma de onda de la señal en el dominio del tiempo.

## Ventaneo y Fragmentación
Ubicación: `signaltools/framing.py`

- `FrameConfig`: Configuración para la extracción de tramas (tamaño, salto, tipo de ventana).
- `frame_signal()`: Divide una señal en fragmentos (frames) solapados. Soporta ventanas: `Hann`, `Hamming`, `Blackman` y `Rectangular`.
- `normalize_signal()` / `standardize_signal()`: Escalado de amplitud y normalización Z-score.

## Extracción de Características (Features)
Ubicación: `signaltools/features.py`

Cálculo de descriptores estadísticos básicos por trama o señal completa:
- **Estadísticas**: Media, mediana, varianza, desviación estándar.
- **Forma**: Skewness (asimetría), Kurtosis (curtosis).
- **Energía**: RMS (Root Mean Square), crest factor, energía total.
- **Cruce por Cero**: `zero_crossing_rate` (ZCR).
- **Dinámica**: `dynamic_range` (rango dinámico en dB) y `peak_to_peak`.

## Detección de Eventos y Picos
Ubicación: `signaltools/detect.py`

- `local_peaks()`: Identifica máximos locales con criterios de altura y distancia mínima.
- `threshold_events()`: Encuentra regiones donde la señal supera un umbral.
- `adaptive_events()`: Detección automática basada en la distribución estadística de la amplitud.
- `onset_strength()`: Calcula el envelope de fuerza de ataque de la señal.
- `anomaly_score()`: Puntuación de anomalía basada en desviaciones estadísticas.

## Fingerprinting (Huellas Digitales)
Ubicación: `signaltools/fingerprint.py`

- `fingerprint_engine()`: Genera un vector de características (`SignalFingerprint`) que resume la identidad técnica de la señal.
- `compare_fingerprints()`: Calcula la similitud entre dos señales mediante distancia Euclidiana o similitud de coseno.
