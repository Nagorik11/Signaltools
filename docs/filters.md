# Módulo Filters (Filtros Digitales)

Este módulo contiene utilidades para el filtrado básico de señales 1D.

## Utilidades de Filtrado 1D
Ubicación: `signaltools/filters.py`

- `moving_average()`: Suavizado por media móvil.
- `median_filter()`: Filtro de mediana para señales 1D.
- `remove_dc()`: Eliminación de componente continua.
- `normalize_peak()`: Normalización por pico máximo.
- `fft_bandpass()`: Filtrado paso-banda directo en el dominio de la frecuencia.

## Diseño de Filtros
Ubicación: `signaltools/filter_design.py`

### Filtros FIR (Respuesta al Impulso Finita)
- `fir_lowpass`, `fir_highpass`, `fir_bandpass`, `fir_bandstop`: Diseño de filtros mediante ventana sinc.
- `fractional_delay_fir`: Filtros para retrasos de tiempo no enteros.
- `savitzky_golay_filter`: Suavizado que preserva picos mediante regresión polinomial.

### Filtros IIR (Respuesta al Impulso Infinita)
- `iir_lowpass_single_pole`: Filtro de polo único simple.
- `biquad_lowpass`, `biquad_highpass`, `biquad_notch`: Secciones bicuadráticas (RBJ) para alta precisión.
- `comb_filter_feedback` / `feedforward`: Filtros de peine para efectos o eliminación de armónicos.

### Filtros Adaptativos y Especiales
- `lms_adaptive_filter`: Implementación básica del algoritmo Least Mean Squares.
- `envelope()`: Extracción de la envolvente de la señal mediante la Transformada de Hilbert.
- `analytic_signal()`: Generación de la señal analítica compleja.

## Filtros de Estado Avanzados
Ubicación: `signaltools/advanced_state_filters.py` y `signaltools/state_filters.py`

- `kalman_filter_1d`: Filtro de Kalman básico.
- `extended_kalman_filter` (EKF) y `unscented_kalman_filter` (UKF): Para estimación en sistemas no lineales.
- `particle_filter_1d`: Filtrado por Monte Carlo secuencial.
- `adaptive_wiener_filter_1d`: Filtrado óptimo de ruido.
