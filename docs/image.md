# Módulo Image (Procesamiento de Imagen y Análisis Forense)

Ubicación: `signaltools/image_decomposition.py` y `signaltools/image_morphology.py`

Este módulo está orientado al análisis de capas de imagen para aplicaciones forenses y mejora de visión.

## Descomposición de Capas
- `decompose_image_layers(image)`: Descompone una imagen en múltiples componentes:
    - Iluminación, Fondo (Background), Primer Plano (Foreground).
    - Textura, Ruido, Bordes.
    - Reflexiones, Sombras y Especulares.
    - Máscaras de tinta y trazos.

## Morfología Matemática (2D y 3D)
Operaciones básicas de procesamiento de imágenes:
- `dilation_2d` / `erosion_2d`: Dilatación y erosión de escala de grises.
- `opening_2d` / `closing_2d`: Apertura y cierre morfológico.
- `median_filter_2d`: Filtro de mediana para eliminación de ruido.
- `morphological_gradient_2d`: Mejora de bordes mediante gradiente.

## Utilidades de Visualización
Ubicación: `signaltools/image_visualization.py`

- `export_comparison_mosaic`: Crea una imagen mosaico comparando las diferentes capas extraídas para su revisión técnica.
