# Learner (learner.py)

El módulo `Learner` proporciona capacidades básicas de aprendizaje automático no supervisado para la clasificación de señales.

## Clase `Learner`

Utiliza el algoritmo **K-Means** de `scikit-learn` para agrupar señales basadas en sus vectores de características (glyphs).

### Métodos Principales

- `collect_features()`: Escanea el directorio de salida en busca de archivos JSON y extrae los vectores de firma.
- `train()`: Entrena el modelo de agrupamiento con los datos recolectados.
- `predict(vector)`: Predice a qué familia (cluster) pertenece un nuevo vector de señal.
