# Módulo Complex DL & AI (Procesamiento Complejo e Inteligencia Artificial)

Este módulo implementa operadores avanzados en el dominio de la frecuencia compleja y herramientas de aprendizaje automático.

## Operadores Tiempo-Frecuencia Complejos
Ubicación: `signaltools/complex_*.py`

- **Atención en TF**: `complex_attention_tf.py` implementa mecanismos de atención acoplados entre múltiples cabezales (heads) que operan sobre espectros complejos.
- **Multicanal**: `complex_multichannel.py` y `complex_frame.py` gestionan STFT (Transformada de Fourier de Tiempo Corto) con soporte para múltiples canales y reconstrucción por solapamiento y suma (Overlap-Add).
- **Operadores Aprendibles**: `complex_learnable_tf.py` permite definir filtros en el dominio de la frecuencia que pueden ser ajustados como parámetros de un modelo.

## Inteligencia Artificial
Ubicación: `signaltools/ai/`

- `Learner`: Implementa un sistema de aprendizaje no supervisado utilizando K-Means.
- **Clasificación de Firmas**: El sistema puede agrupar señales en familias basadas en sus vectores de características extraídos por el módulo `bridge`.
- **Entrenamiento**: `train()` permite procesar todos los resultados previos para encontrar patrones comunes en la base de datos de señales analizadas.
