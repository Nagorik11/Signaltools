# Muestras (Samples)

El directorio de muestras contiene archivos de ejemplo para probar las funcionalidades del sistema sin necesidad de fuentes externas.

## Archivos de Señal
Ubicación: `/samples/`

- `heartbeat.json`: Una señal grabada (o simulada) de un latido cardíaco, ideal para probar la detección de picos (`local_peaks`) y el análisis de tono.
- `sample_signal.json`: Señal genérica con ruido para validación de filtros y análisis espectral.

## Imágenes
- `imagen.png` (en la raíz): Imagen de prueba para las herramientas de descomposición de capas forenses.

## Uso de Muestras
Puedes cargar estas muestras usando el `Ingestor`:
```python
from signaltools.io import Ingestor
signal_data = Ingestor.from_json("samples/heartbeat.json")
```
O directamente vía CLI:
```bash
python -m signaltools samples/heartbeat.json --output analysis.json
```
