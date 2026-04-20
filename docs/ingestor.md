# Ingestor (ingestor.py)

El `Ingestor` es el componente encargado de la carga y validación inicial de los datos de señal desde diferentes fuentes.

## Formatos Soportados

- **JSON**: Archivos con estructuras de lista o diccionario.
- **Texto (.txt, .log, .md)**: Archivos de texto plano con valores numéricos.
- **Binario**: Carga directa de flujos de bytes.
- **WAV**: A través de las utilidades de `io.wav`.

## Funciones Principales

- `Ingestor.from_json(path)`: Carga datos desde un archivo JSON.
- `Ingestor.from_text(path)`: Parsea un archivo de texto buscando números.
- `Ingestor.from_binary(path)`: Lee bytes directamente y los convierte a float64.
