# Documentación de SignalTools (Español)

Esta carpeta contiene la documentación técnica del proyecto `signaltools`, organizada por módulos.

## Índice de Contenidos

- [Referencia de API](api_reference.md): Lista completa de funciones y clases.
- [Ingestor](ingestor.md): Carga y validación de datos iniciales.
- [I/O (E/S)](io.md): Manejo de archivos y WAV.
- [Spectral (Espectral)](spectral.md): Transformada de Fourier y métricas de frecuencia.
- [Features (Características)](features.md): Descriptores estadísticos y de energía.
- [Framing (Ventaneo)](framing.md): Preparación de tramas y normalización.
- [Detect (Detección)](detect.md): Encuentro de eventos, picos y anomalías.
- [Bitlayer](bitlayer.md): Análisis de la estructura binaria.
- [Bridge (Puente)](bridge.md): Firmas de señales y vectores glyph.
- [Learner](learner.md): Aprendizaje automático y K-Means.
- [Filters (Filtros)](filters.md): Filtrado básico de señales.
- [Wavelets](wavelets.md): Descomposición y reconstrucción avanzada (1D a 5D).
- [Forensics (Forense)](forensics.md): Análisis pericial, cadena de custodia y sellos de tiempo.
- [Graphs (Grafos)](graphs.md): Procesamiento de señales en grafos.
- [Complex DL & AI](complex_dl.md): Operadores complejos e inteligencia artificial.
- [Utils (Utilidades)](utils.md): Logging y excepciones.

## Recursos del Proyecto

- [Tests (Pruebas)](tests.md): Suite de pruebas automatizadas y cobertura.
- [Benchmarks (Rendimiento)](benchmarks.md): Mediciones de velocidad y eficiencia.
- [Samples (Muestras)](samples.md): Ejemplos de señales e imágenes de prueba.
- [Infraestructura](infra.md): Configuración del proyecto, dependencias y estructura.

## Otros Componentes

- **CLI (Interfaz de Línea de Comandos)**: Se puede ejecutar el análisis avanzado directamente desde la terminal usando `python -m signaltools`. Ejecuta `python -m signaltools --help` para ver todas las opciones disponibles, incluyendo el modo forense.
- **Manager**: La clase `Manager` en `manager.py` proporciona una interfaz simplificada para usuarios que vienen de versiones anteriores del proyecto.
