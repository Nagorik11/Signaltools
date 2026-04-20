# IO & Buffer Management

Módulo encargado de la lectura y escritura de archivos de datos y audio.

## Funciones Principales

### `read_audio_file(path, sample_rate=44100)`
Utiliza FFmpeg para decodificar cualquier formato de audio (MP3, WAV, MP4) a un buffer de bytes raw (PCM 16-bit).

### `write_wav(path, signal, sample_rate=44100)`
Guarda una lista de valores numéricos como un archivo WAV mono de 16 bits. Escala automáticamente la señal basándose en el valor máximo.

### `read_signal_file(path)`
Lectura genérica de archivos binarios en un objeto `SignalBuffer`.

### `guess_numeric_views(raw_bytes)`
Intenta interpretar un buffer de bytes como diferentes tipos de datos numéricos (uint8, int16, float32).

## Clases

### `SignalBuffer`
Contenedor para datos raw que incluye la ruta del archivo y métodos de previsualización hexadecimal.