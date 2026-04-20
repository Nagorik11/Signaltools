# Procesamiento de Grafos Avanzado (Detalle Técnico)

Documentación técnica sobre el análisis de señales en grafos y arquitecturas de aprendizaje profundo sobre grafos.

## Wavelets de Grafo (`graph_wavelets.py`)
Implementación de la Transformada Wavelet en Grafos (SGWT).
- `chebyshev_graph_filter`: Aproximación eficiente de filtros espectrales en grafos mediante polinomios de Chebyshev.
- `graph_wavelet_decompose`: Descomposición multiescala de señales definidas sobre nodos.

## Posicionamiento y Estructura (`graph_positional.py`)
- `laplacian_positional_encoding`: Extracción de vectores de posición basados en los vectores propios del Laplaciano (similar al posicionamiento en modelos Transformer).

## Filtros Profundos (GNN) (`graph_deep_filters.py`)
Capas neuronales especializadas en señales estructuradas:
- `EdgeConditionedConv`: Convoluciones donde los pesos dependen de los atributos de las aristas.
- `GraphAttentionLayer`: Mecanismo de auto-atención para grafos.
- `DiffPoolLayer`: Operación de agrupación (pooling) diderenciable para arquitecturas jerárquicas.
