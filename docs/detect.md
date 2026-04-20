# Detección y Eventos (detect.py)

El módulo `detect` proporciona algoritmos para identificar eventos y anomalías en los datos de la señal.

## Funciones

### `threshold_events(signal: list[float], threshold: float) -> list[dict]`
Detecta regiones continuas donde la magnitud absoluta de la señal supera un umbral.
Retorna una lista de diccionarios de eventos con `start` (inicio), `end` (fin) y `length` (longitud).

### `local_peaks(signal: list[float], min_height: float = 0.0) -> list[dict]`
Identifica máximos locales en la señal que son mayores o iguales a `min_height`.

### `anomaly_score(signal: list[float]) -> list[float]`
Calcula un vector de puntuación de anomalías basado en el Z-score para la señal.
