# Módulo Utils (Utilidades y Sistema)

Este módulo contiene componentes transversales utilizados por todo el paquete `signaltools`.

## Logging (Registro)
Ubicación: `signaltools/logging_utils.py`

- `configure_logging(level)`: Configura el sistema de logs a nivel global del paquete.
- `get_logger(name)`: Obtiene un logger con el espacio de nombres adecuado.

## Excepciones
Ubicación: `signaltools/exceptions.py`

- `SignalToolsError`: Excepción base de la que heredan todos los errores del paquete.
- `SignalValidationError`: Lanzada cuando los parámetros de entrada o los datos de la señal no cumplen con los requisitos técnicos.

## Funciones de Utilidad y Validación
Ubicación: `signaltools/utils.py`

- `to_1d_float_array(signal)`: Convierte y valida cualquier entrada (listas, arrays) a un array de NumPy 1D de tipo float64.
- `ensure_positive_int()` / `ensure_non_negative_float()`: Helpers para validación de parámetros de diseño de filtros.
- `safe_mean()`: Calcula la media de forma segura, evitando errores por listas vacías.
