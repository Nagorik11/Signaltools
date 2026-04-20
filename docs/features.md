# Características (features.py)

El módulo `features` contiene funciones para extraer descriptores estadísticos y de energía de señales unidimensionales.

## Funciones

### `rms(signal)`
Calcula el valor eficaz (Root Mean Square) de la señal.

### `zero_crossing_rate(signal)`
Calcula la tasa de cruce por cero, útil para identificar el contenido de ruido o ruidos de alta frecuencia.

### `crest_factor(signal)`
Calcula la relación entre la magnitud pico y la energía RMS.

### `first_derivative(signal)` / `second_derivative(signal)`
Calcula las derivadas numéricas de la señal.

### `frame_feature_vector(frame: list[float]) -> dict`
Retorna un diccionario que contiene todas las características básicas para una trama (frame) dada.
