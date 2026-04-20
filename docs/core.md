# Módulo Core (Núcleo)

Este módulo define la base del sistema para el manejo y análisis de señales.

## Clase `Signal`
Ubicación: `signaltools/core/signal.py`

La clase `Signal` es el contenedor principal de datos de señal y proporciona métodos para operaciones básicas.

### Métodos Principales
- `normalize()`: Aplica eliminación de componente de continua (DC) y normaliza la señal.
- `extract_features()`: Extrae vectores de características como media, varianza y RMS.
- `get_bit_layer()`: Realiza un análisis de entropía de bits sobre los datos brutos.
- `get_glyph_vector()`: Genera un vector numérico que representa la "firma" de la señal.

---

## Clase `SignalAnalyzer`
Ubicación: `signaltools/core/analyzer.py`

Esta clase se encarga de realizar análisis segmentados de una instancia de `Signal`.

### Métodos Principales
- `get_timeline_analysis()`: Divide la señal en ventanas y analiza energía, planicidad espectral y complejidad para cada segmento.
- `generate_summary()`: Proporciona estadísticas globales (media, máximo, varianza) sobre el análisis temporal.
