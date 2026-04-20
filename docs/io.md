# Entrada y Salida (I/O)

Manejo de archivos y buffers de datos para `signaltools`.

## Funciones

### `read_wav(path)`
Lee archivos de audio en formato WAV y los convierte en una lista de flotantes normalizados.

### `read_signal_file(path)`
Lectura genérica de archivos binarios en un objeto `SignalBuffer`.

### `guess_numeric_views(raw_bytes)`
Intenta interpretar un buffer de bytes como diferentes tipos de datos numéricos (uint8, int16, float32).

## Clases

### `SignalBuffer`
Contenedor para datos en bruto que incluye la ruta del archivo y métodos de previsualización hexadecimal.
