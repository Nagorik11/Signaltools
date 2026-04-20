# signaltools.bitlayer

Documentación del módulo `signaltools.bitlayer`.

Resumen
- Herramientas para convertir bytes a bits, analizar propiedades estadísticas de la secuencia de bits y generar una "firma" compacta.
- Útil para inspección rápida de streams binarios, detección de periodicidad y generación de representaciones legibles.

Archivo fuente
- signaltools/bitlayer.py

API (funciones y clases principales)

- bytes_to_bits(raw: bytes) -> list[int]
  - Convierte un objeto `bytes` a una lista de bits (orden MSB primero por byte).
  - Retorna: lista de 0/1.
  - Complejidad: O(n) en número de bytes.

- bits_to_signal(bits: list[int]) -> list[float]
  - Convierte bits a una señal bipolar: 1 -> 1.0, 0 -> -1.0.
  - Uso: visualización o procesamiento como señal continua.

- bit_entropy(bits: list[int]) -> float
  - Entropía de Shannon binaria de la secuencia.
  - Retorna 0.0 para lista vacía.
  - Nota: maneja probabilidades 0/1 sin errores.

- bit_transitions(bits: list[int]) -> int
  - Cuenta el número de transiciones (0→1 o 1→0) adyacentes.
  - Útil para densidad de cambios.

- run_lengths(bits: list[int]) -> list[tuple[int,int]]
  - Devuelve lista de tuplas (valor_bit, longitud) con las corridas consecutivas.
  - Ejemplo: [1,1,0,0,0,1] -> [(1,2),(0,3),(1,1)]

- detect_period(bits: list[int], max_period: int = 64) -> list[tuple[int,float]]
  - Busca periodos dominantes comparando igualdad desplazada.
  - Retorna los 5 periodos mejor puntuados (periodo, puntuación de coincidencia).
  - Nota: uso conservador de `max_period` por coste computacional.

- bit_balance(bits: list[int]) -> float
  - Desbalance absoluto entre 1s y 0s normalizado en [0,1].
  - 0 indica perfecto equilibrio (misma cantidad de 0 y 1).

- average_run_length(runs: list[tuple[int,int]]) -> float
  - Longitud promedio de las corridas (usa salida de run_lengths).

- longest_run(runs: list[tuple[int,int]]) -> int
  - Longitud máxima de corrida.

- BitSignature (dataclass)
  - Campos:
    - total_bits: int
    - entropy: float
    - transitions: int
    - transition_density: float
    - avg_run_length: float
    - longest_run: int
    - balance: float
    - dominant_periods: list[tuple[int,float]]
    - meta: dict
  - Método: to_dict() para serializar.

- build_bit_signature(bits: list[int]) -> BitSignature
  - Construye la firma completa: resume las métricas, calcula corridas y detecta periodos sobre una muestra (hasta 10000 bits).
  - Optimización: muestra periódica para streams grandes.

- compact_bit_expression(sig: BitSignature) -> str
  - Genera una representación compacta tipo:
    BIT{N:1024|H:0.999|T:512|D:0.500|R:2.00|L:50|B:0.02|P:[3:0.80,7:0.50,...]}

- analyze_bitlayer(raw: bytes) -> dict
  - Pipeline completo: bytes -> bits -> preview de señal (128 muestras) -> firma -> expresión compacta.
  - Retorna:
    - bits: primeras 128 bits
    - signal_preview: primerias 128 muestras bipolar
    - signature: firma serializada (dict)
    - compact: expresión compacta

Ejemplos de uso

- Análisis rápido de un buffer:
```python
from signaltools.bitlayer import analyze_bitlayer
res = analyze_bitlayer(b"\x00\xff\x0f\xf0")
print(res["compact"])
```

- Obtener la firma detallada:
```python
from signaltools.bitlayer import bytes_to_bits, build_bit_signature
bits = bytes_to_bits(my_bytes)
sig = build_bit_signature(bits)
print(sig.to_dict())
```

Consideraciones y buenas prácticas
- Las funciones asumen entrada limpia; convertir primero a `bytes` o `list[int]` según corresponda.
- Detectar período es costoso para secuencias largas; `build_bit_signature` aplica muestreo (hasta 10k) para limitar coste.
- Operaciones numéricas y estadísticas usan lógica defensiva (ej.: protección contra divisiones por cero).
- Para análisis a gran escala, procesar por bloques y agregar métricas incrementales.

Notas de implementación
- Orden de bits por byte: MSB primero ((byte >> (7-i)) & 1).
- La salida de `analyze_bitlayer` incluye solo un preview de 128 muestras para evitar grandes cargas en memoria cuando se serializa.

Licencia y mantenimiento
- El módulo forma parte del paquete `signaltools`. Revisar pruebas y casos límite si se extiende a flujos muy grandes o formatos encapuchados.

```// filepath: /home/kali/Escritorio/signaltools_src_layout_project/src/docs/bitlayer.md
# signaltools.bitlayer

Documentación del módulo `signaltools.bitlayer`.

Resumen
- Herramientas para convertir bytes a bits, analizar propiedades estadísticas de la secuencia de bits y generar una "firma" compacta.
- Útil para inspección rápida de streams binarios, detección de periodicidad y generación de representaciones legibles.

Archivo fuente
- signaltools/bitlayer.py

API (funciones y clases principales)

- bytes_to_bits(raw: bytes) -> list[int]
  - Convierte un objeto `bytes` a una lista de bits (orden MSB primero por byte).
  - Retorna: lista de 0/1.
  - Complejidad: O(n) en número de bytes.

- bits_to_signal(bits: list[int]) -> list[float]
  - Convierte bits a una señal bipolar: 1 -> 1.0, 0 -> -1.0.
  - Uso: visualización o procesamiento como señal continua.

- bit_entropy(bits: list[int]) -> float
  - Entropía de Shannon binaria de la secuencia.
  - Retorna 0.0 para lista vacía.
  - Nota: maneja probabilidades 0/1 sin errores.

- bit_transitions(bits: list[int]) -> int
  - Cuenta el número de transiciones (0→1 o 1→0) adyacentes.
  - Útil para densidad de cambios.

- run_lengths(bits: list[int]) -> list[tuple[int,int]]
  - Devuelve lista de tuplas (valor_bit, longitud) con las corridas consecutivas.
  - Ejemplo: [1,1,0,0,0,1] -> [(1,2),(0,3),(1,1)]

- detect_period(bits: list[int], max_period: int = 64) -> list[tuple[int,float]]
  - Busca periodos dominantes comparando igualdad desplazada.
  - Retorna los 5 periodos mejor puntuados (periodo, puntuación de coincidencia).
  - Nota: uso conservador de `max_period` por coste computacional.

- bit_balance(bits: list[int]) -> float
  - Desbalance absoluto entre 1s y 0s normalizado en [0,1].
  - 0 indica perfecto equilibrio (misma cantidad de 0 y 1).

- average_run_length(runs: list[tuple[int,int]]) -> float
  - Longitud promedio de las corridas (usa salida de run_lengths).

- longest_run(runs: list[tuple[int,int]]) -> int
  - Longitud máxima de corrida.

- BitSignature (dataclass)
  - Campos:
    - total_bits: int
    - entropy: float
    - transitions: int
    - transition_density: float
    - avg_run_length: float
    - longest_run: int
    - balance: float
    - dominant_periods: list[tuple[int,float]]
    - meta: dict
  - Método: to_dict() para serializar.

- build_bit_signature(bits: list[int]) -> BitSignature
  - Construye la firma completa: resume las métricas, calcula corridas y detecta periodos sobre una muestra (hasta 10000 bits).
  - Optimización: muestra periódica para streams grandes.

- compact_bit_expression(sig: BitSignature) -> str
  - Genera una representación compacta tipo:
    BIT{N:1024|H:0.999|T:512|D:0.500|R:2.00|L:50|B:0.02|P:[3:0.80,7:0.50,...]}

- analyze_bitlayer(raw: bytes) -> dict
  - Pipeline completo: bytes -> bits -> preview de señal (128 muestras) -> firma -> expresión compacta.
  - Retorna:
    - bits: primeras 128 bits
    - signal_preview: primerias 128 muestras bipolar
    - signature: firma serializada (dict)
    - compact: expresión compacta

Ejemplos de uso

- Análisis rápido de un buffer:
```python
from signaltools.bitlayer import analyze_bitlayer
res = analyze_bitlayer(b"\x00\xff\x0f\xf0")
print(res["compact"])
```

- Obtener la firma detallada:
```python
from signaltools.bitlayer import bytes_to_bits, build_bit_signature
bits = bytes_to_bits(my_bytes)
sig = build_bit_signature(bits)
print(sig.to_dict())
```

