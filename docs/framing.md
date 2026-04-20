# Framing & Normalization

Herramientas para la preparación de señales antes del análisis espectral o de características.

## Configuración

Utiliza la dataclass `FrameConfig` para definir:
- `frame_size`: Tamaño de la ventana (defecto: 256).
- `hop_size`: Salto entre ventanas (defecto: 128).

## Funciones

### `frame_signal(signal, cfg)`
Divide una señal unidimensional en una lista de frames (ventanas) solapadas.

### `normalize_signal(signal)`
Escala la señal al rango [-1.0, 1.0] basándose en el valor absoluto máximo.

### `detrend_mean(signal)`
Elimina la componente de corriente continua (DC offset) restando la media de la señal.