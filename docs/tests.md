# Pruebas (Tests) del Proyecto

Este proyecto cuenta con una suite de pruebas exhaustiva para verificar la integridad de todos los algoritmos de procesamiento de señales.

## Organización de las Pruebas
Ubicación: `/tests/`

Las pruebas están divididas en categorías que reflejan la complejidad del sistema:

### Pruebas de Núcleo y API
- `test_public_api.py`: Verifica que todos los módulos exportados estén disponibles.
- `test_signal_ops_more.py`: Pruebas de operaciones básicas sobre la clase `Signal`.
- `test_io_core_cli.py`: Validación de la ingesta de archivos y la interfaz de comandos.

### Pruebas de Procesamiento Avanzado
- `test_filter_design.py`: Pruebas de filtros FIR e IIR.
- `test_wavelet_extended_attention_gnn.py`: Pruebas de la integración entre Wavelets y Redes Neuronales de Grafos.
- `test_image_decomposition.py` / `test_image_forensics.py`: Validación del pipeline de análisis de imágenes.
- `test_graph_transformer_multichannel_cdf.py`: Pruebas de procesadores de grafos complejos.

### Pruebas de Escenarios 5D y Complejos
- `test_complex_frame_recurrent_5d.py`: Validación de tensores de 5 dimensiones.
- `test_longmemory_branch_regularized.py`: Pruebas de estabilidad en operadores con memoria de largo alcance.

## Ejecución de Pruebas
Para ejecutar todas las pruebas, utiliza el comando:
```bash
pytest
```
El reporte de cobertura se genera en `coverage.xml`.
