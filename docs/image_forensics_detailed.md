# Análisis Forense de Imagen Detallado (Detalle Técnico)

Herramientas para la generación de reportes periciales y visualización de descomposiciones de imagen.

## Pipeline Forense de Imagen (`image_forensics.py`)
- `forensic_decompose_image()`: Orquestador principal que integra la descomposición de capas con el sistema de trazabilidad forense.
- **Funcionalidades**:
    - Generación de hashes para cada capa extraída.
    - Creación de un paquete (bundle) que incluye manifiesto, cadena de custodia y sellado de tiempo.
    - Exportación de reportes en formato JSON compatibles con estándares de auditoría.

## Visualización y Exportación (`image_visualization.py`)
- `ComparisonMosaicResult`: Estructura que almacena rutas y metadatos de mosaicos comparativos.
- `export_comparison_mosaic()`: Genera una imagen que muestra la imagen original junto a sus capas de iluminación, ruido, textura y bordes para facilitar la inspección visual.

## Morfología Detallada
Ubicación: `image_morphology.py`
- Soporte para operaciones 2D y 3D (para volúmenes de datos).
- Operaciones de gradiente morfológico y "top-hat" para resaltar detalles finos o corregir iluminación no uniforme.
