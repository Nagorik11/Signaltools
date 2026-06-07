# Módulo ProcessingPipeline (Flujos de Procesamiento Personalizables)

Este módulo proporciona un sistema flexible para construir y ejecutar pipelines de procesamiento de señales.

## Clases Principales

### `ProcessingPipeline`

Clase principal para construir pipelines personalizados de procesamiento de señales.

**Características:**
- Encadenamiento fluido de pasos de procesamiento
- Contexto compartido entre pasos
- Manejo robusto de errores
- Ejecución paralela opcional
- Serialización a/desde JSON

**Ejemplo básico:**

```python
from signaltools import ProcessingPipeline
from signaltools.framing import normalize_signal, detrend_mean

# Crear pipeline
pipeline = ProcessingPipeline(name="mi_pipeline")

# Añadir pasos
pipeline.add_step("normalize", normalize_signal)
pipeline.add_step("detrend", detrend_mean)

# Ejecutar
signal = [1.0, 2.0, 3.0, 4.0, 5.0]
result = pipeline.execute(signal)

print(f"Éxito: {result.success}")
print(f"Tiempo total: {result.total_time:.4f}s")
```

### `PipelineStep`

Representa un paso individual en el pipeline.

**Atributos:**
- `name`: Identificador único del paso
- `func`: Función a ejecutar
- `args`: Argumentos posicionales
- `kwargs`: Argumentos keyword
- `description`: Descripción opcional

### `PipelineResult`

Contiene el resultado de ejecutar un pipeline.

**Atributos:**
- `output`: Datos de salida del último paso
- `context`: Contexto compartido
- `step_results`: Resultados por paso
- `total_time`: Tiempo total de ejecución
- `success`: Estado de éxito/fallo

**Métodos:**
- `to_dict()`: Convierte a diccionario
- `save(filepath)`: Guarda resultado en JSON

## Métodos de ProcessingPipeline

### Gestión de Pasos

```python
# Añadir paso
pipeline.add_step("nombre", funcion, arg1, arg2, kwarg1=value)

# Insertar en posición específica
pipeline.insert_step(0, "primer_paso", funcion)

# Eliminar paso por nombre
pipeline.remove_step("nombre")

# Limpiar todos los pasos
pipeline.clear_steps()
```

### Ejecución

```python
# Ejecución simple
result = pipeline.execute(datos)

# Con contexto personalizado
context = {"parametro": "valor"}
result = pipeline.execute(datos, context=context)

# Continuar tras error
result = pipeline.execute(datos, stop_on_error=False)

# Ejecución en batch
results = pipeline.execute_batch([datos1, datos2, datos3])

# Ejecución paralela
results = pipeline.execute_batch([datos1, datos2], parallel=True, max_workers=4)
```

### Serialización

```python
# Guardar configuración
pipeline.save_config("config.json")

# Cargar desde configuración
loaded_pipeline = ProcessingPipeline.load_config("config.json")

# Exportar a diccionario
config = pipeline.to_config()

# Importar desde diccionario
new_pipeline = ProcessingPipeline.from_config(config)
```

## Funciones Convenientes

### `create_preprocessing_pipeline()`

Crea un pipeline de preprocesamiento estándar.

```python
from signaltools import create_preprocessing_pipeline

pipeline = create_preprocessing_pipeline(
    normalize=True,
    detrend=True,
    filter_params={"cutoff": 0.1}
)
```

### `create_analysis_pipeline()`

Crea un pipeline de análisis estándar.

```python
from signaltools import create_analysis_pipeline

pipeline = create_analysis_pipeline(sample_rate=44100)
```

## Ejemplos Avanzados

### Pipeline con Parámetros

```python
from signaltools import ProcessingPipeline
from signaltools.filters import lowpass_filter

pipeline = ProcessingPipeline()
pipeline.add_step(
    "filtro",
    lowpass_filter,
    cutoff=0.1,
    order=4,
    description="Filtro pasa-bajos"
)
```

### Pipeline con Contexto Compartido

```python
def step_with_context(data, _context):
    # Acceder a contexto compartido
    previous_result = _context.get('previous_step')
    # Modificar contexto
    _context['current_result'] = data
    return data

pipeline = ProcessingPipeline()
pipeline.add_step("paso1", step_with_context, _context=True)
```

### Batch Paralelo

```python
signals = [generate_signal() for _ in range(10)]

results = pipeline.execute_batch(
    signals,
    parallel=True,
    max_workers=4
)

for i, result in enumerate(results):
    print(f"Señal {i}: {'OK' if result.success else 'FALLÓ'}")
```

### Guardar Resultados

```python
result = pipeline.execute(signal)

# Guardar resultados completos
result.save("output/resultados.json")

# Acceder a metadatos
print(result.step_results)
print(result.context['total_time'])
```

## Manejo de Errores

```python
# Parar al primer error (por defecto)
result = pipeline.execute(datos, stop_on_error=True)

# Continuar y reportar errores
result = pipeline.execute(datos, stop_on_error=False)

# Verificar errores por paso
for step_name, step_result in result.step_results.items():
    if step_result['status'] == 'failed':
        print(f"Paso {step_name} falló: {step_result.get('error')}")
```

## Mejores Prácticas

1. **Nombres únicos**: Asegúrate de que cada paso tenga un nombre único
2. **Funciones puras**: Usa funciones sin efectos secundarios cuando sea posible
3. **Contexto ligero**: Mantén el contexto compartido pequeño y serializable
4. **Validación temprana**: Valida parámetros al añadir pasos, no al ejecutar
5. **Logging**: Revisa los logs para depurar problemas de ejecución

## Integración con Otras Herramientas

El pipeline se integra perfectamente con otros módulos de signaltools:

```python
from signaltools import ProcessingPipeline
from signaltools.framing import normalize_signal
from signaltools.detect import adaptive_events
from signaltools.spectral import spectral_centroid

pipeline = ProcessingPipeline(name="analisis_completo")
pipeline.add_step("normalize", normalize_signal)
pipeline.add_step("events", adaptive_events)
# ... más pasos

result = pipeline.execute(mis_datos)
```
