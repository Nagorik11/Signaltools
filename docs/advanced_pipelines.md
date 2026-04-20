# Módulo Advanced Pipelines (Flujos de Análisis Avanzados)

Este módulo orquestra múltiples técnicas para realizar un escaneo profundo de las señales.

## Pipeline de Orquestación
Ubicación: `signaltools/pipeline.py`

- `analyze_signal_advanced()`: La función principal de orquestación. Ejecuta análisis temporal, espectral, simbólico, de bits y genera la huella digital en un solo paso.
- `AdvancedSignalAnalysis`: Clase que estructura todos los resultados (resumen, diagnósticos, temporal, espectral).

## Puentes Simbólicos y Capas
Ubicación: `signaltools/bridge.py`

- `analyze_signal_layered()`: Descompone la señal en capas abstractas (tonal, ruido, transitorios).
- `signal_signature()`: Extrae un "glyph vector" único que puede ser utilizado para clasificación o búsqueda.

## Análisis de Capa de Bits
Ubicación: `signaltools/bitlayer.py`

Analiza la señal como un flujo de datos binarios:
- `analyze_bitlayer()`: Calcula la entropía de bits, redundancia y patrones de distribución en el flujo de bytes subyacente.

## Modulación y Multitasa
- `modulate.py`: Funciones para Modulación de Amplitud (AM) y Frecuencia (FM).
- `multirate.py`: Implementa `decimate` (diezmar), `interpolate` (interpolar) y bancos de filtros de dos bandas mediante polifase.
