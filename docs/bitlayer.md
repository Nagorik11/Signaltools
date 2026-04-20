# Análisis de Capa de Bits (bitlayer.md)

Análisis de la estructura binaria subyacente de la información.

## Métricas de Bits

- **Entropía**: Medida de incertidumbre/información en el flujo de bits.
- **Densidad de Transición**: Frecuencia con la que los bits cambian de 0 a 1 o viceversa.
- **Balance**: Relación entre la cantidad de ceros y unos.

## Funciones Clave

### `analyze_bitlayer(raw_bytes)`
Realiza un análisis completo y retorna un diccionario con la firma de bits y una previsualización.

### `compact_bit_expression(sig)`
Genera una representación en cadena (String) legible para humanos que resume las propiedades de los bits.
Ejemplo: `BIT{N:128|H:0.950|...}`
