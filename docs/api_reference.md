# Referencia de la API (API_REFERENCE)

Esta es una referencia completa de las funciones y clases exportadas por el paquete `signaltools`.

## Núcleo (Core) - `core/`
- `Signal`: Clase base para contenedores de señales.
- `SignalAnalyzer`: Clase para realizar análisis por ventanas sobre un objeto `Signal`.

## Entrada/Salida (I/O) - `io/`
- `Ingestor`: Utilidad para cargar señales desde diversos formatos (JSON, Texto, Binario).
- `read_wav(path)` / `write_wav(path, data, sample_rate)`: Manejo de archivos WAV.
- `read_signal_file(path)`: Función de conveniencia para cargar señales.

## Espectral - `spectral.py`
- `dft(signal, mode="magnitude")`: Transformada Discreta de Fourier.
- `stft(signal, frame_size, hop_size)`: Transformada de Fourier de Tiempo Corto.
- `spectrogram_matrix(...)`: Genera la matriz del espectrograma.
- `spectral_centroid(signal, sample_rate)`: Centroide espectral.
- `spectral_bandwidth(signal, sample_rate)`: Ancho de banda espectral.
- `spectral_rolloff(signal, sample_rate)`: Frecuencia de rolloff espectral.
- `estimate_pitch(signal, sample_rate)`: Estimación de tono fundamental.

## Análisis Forense de Imagen - `image_forensics.py`
- `forensic_decompose_image(source, ...)`: Pipeline completo de descomposición y auditoría forense.
- `save_decomposition_layers(...)`: Exportación de capas a disco.

(Para una lista completa de funciones de imagen, ver `image.md` o el código fuente).

## Utilidades y Filtros
- `filters.py`: Filtros de suavizado y básicos.
- `filter_design.py`: Diseño de filtros FIR/IIR.
- `utils.py`: Validaciones y conversiones de tipos.
- `logging_utils.py`: Configuración del sistema de logs.
