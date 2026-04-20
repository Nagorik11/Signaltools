# Módulo Graphs (Procesamiento de Señales en Grafos - GSP)

Ubicación: `signaltools/graph_filters.py` y módulos `graph_*`.

Este módulo permite aplicar conceptos de procesamiento de señales a datos estructurados en grafos, utilizando el Laplaciano del grafo.

## Fundamentos de Grafos
- `graph_laplacian(adjacency)`: Calcula la matriz Laplaciana (combinatoria o normalizada) a partir de una matriz de adyacencia.
- `graph_fourier_basis()`: Calcula los valores y vectores propios del Laplaciano, permitiendo el análisis en el "dominio de la frecuencia del grafo".

## Filtrado en Grafos
- `graph_filter_signal()`: Filtra una señal definida sobre los nodos del grafo aplicando una respuesta en frecuencia específica.
- `graph_polynomial_filter()`: Aplica filtros mediante polinomios del Laplaciano, evitando la descomposición en valores propios (más eficiente para grafos grandes).

## Herramientas Avanzadas (Módulos Deep/GNN)
Ubicación: `signaltools/graph_deep_filters.py`

Implementación de arquitecturas modernas para señales en grafos:
- Capas de Convolución Condicionada por Aristas (`EdgeConditionedConv`).
- Transformadores de Grafos con atención QKV.
- Memoria con Puertas Híbrida Temporal-Grafo.
- Redes Neuronales de Grafos (GNN) profundas y jerárquicas.
