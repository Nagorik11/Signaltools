# Puente y Firma (bridge.py)

El módulo `bridge` conecta características de bajo nivel con descriptores y representaciones de alto nivel.

## Clases

### `SignalSignature`
Una dataclass completa que captura todo el estado del análisis de una señal.
Incluye dimensiones, promedio de características por trama, propiedades espectrales, conteo de eventos y derivadas.

## Funciones

### `signal_signature(...) -> SignalSignature`
Genera una `SignalSignature` completa a partir de una señal en bruto. Esta es la API principal de alto nivel para la caracterización de señales.

### `signature_to_glyph_vector(sig: SignalSignature) -> list[float]`
Comprime una `SignalSignature` en una lista plana de números flotantes, ideal para comparación por similitud, agrupamiento (clustering) o visualización.
