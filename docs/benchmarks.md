# Benchmarks (Rendimiento)

Este módulo permite medir y comparar el rendimiento de los diferentes algoritmos de procesamiento, especialmente útil tras optimizaciones.

## Herramientas de Medición
Ubicación: `/benchmarks/`

- `run_benchmarks.py`: Script principal para ejecutar las pruebas de velocidad y uso de memoria.
- `benchmark_results.json`: Almacena los resultados históricos para comparar la evolución del rendimiento.

## Qué se mide
- **Tiempo de Ejecución**: Segundos requeridos para descomposiciones complejas (Wavelets 5D, GNN).
- **Consumo de Memoria**: Pico de RAM utilizado durante el procesamiento de señales grandes o imágenes.
- **Latencia de Pipeline**: Tiempo total que tarda una señal en pasar por todo el flujo de análisis avanzado.

## Cómo ejecutar los Benchmarks
Se recomienda ejecutar los benchmarks antes de realizar cambios importantes en la arquitectura:
```bash
python benchmarks/run_benchmarks.py
```
