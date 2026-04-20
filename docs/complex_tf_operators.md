# Operadores de Tiempo-Frecuencia Complejos (Detalle Técnico)

Este documento detalla los operadores que manipulan señales en el dominio de la STFT compleja, permitiendo transformaciones no lineales y mecanismos de atención.

## Operadores Aprendibles (`complex_learnable_tf.py`)
Permite aplicar transformaciones afines complejas $Y = X \cdot W + b$ seguidas de una fase de rotación y activación.
- `complex_learnable_tf_operator`: Aplica ganancia, sesgo (bias) y rotación de fase.
- **Activaciones Soportadas**: `tanh`, `sigmoid_mag`, `relu_mag`.
- **Arquitectura Residual**: Opción de sumar la entrada a la salida transformada.

## Atención Acoplada (`complex_attention_tf.py`)
Implementa mecanismos de atención entre múltiples "cabezales" en el dominio tiempo-frecuencia.
- `complex_multiband_head_coupling_operator`: Acopla cabezales mediante una matriz de peso.
- `temporal_complex_head_coupling_operator`: Permite que el acoplamiento varíe en cada trama temporal.
- `long_memory_temporal_head_coupling`: Incorpora memoria con decaimiento exponencial para capturar dependencias de largo alcance.

## Procesamiento Multicanal (`complex_multichannel.py` / `complex_frame.py`)
Gestión de señales complejas distribuidas en múltiples canales.
- `complex_stft_multichannel`: Calcula la STFT de forma independiente para cada canal.
- `complex_frame_operator`: Aplica máscaras espectrales y reconstruye la señal mediante Overlap-Add.

## Multi-Head TF (`complex_multihead_tf.py`)
Divide el espectro en bandas de frecuencia y asigna cada una a un cabezal de procesamiento independiente.
- `multihead_band_complex_tf_operator`: Distribución de bandas y procesamiento paralelo.

## Espectral Complejo (`complex_spectral.py`)
Herramientas para la manipulación directa de magnitudes y fases complejas en el dominio de la frecuencia.
