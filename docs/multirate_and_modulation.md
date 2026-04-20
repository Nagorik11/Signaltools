# Procesamiento Multitasa y Modulación (Detalle Técnico)

Este documento cubre las técnicas de cambio de frecuencia de muestreo y síntesis de señales mediante modulación.

## Procesamiento Multitasa (`multirate.py`)
- `decimate()` / `interpolate()`: Cambio de tasa de muestreo con filtrado anti-alias integrado.
- `TwoBandAnalysisBank`: Divide la señal en componentes de baja y alta frecuencia (análisis/síntesis).
- `polyphase_decompose()`: Descomposición polifase de coeficientes de filtro para implementaciones eficientes de bancos de filtros.

## Modulación de Señal (`modulate.py`)
- `amplitude_modulation` (AM): Modulación de amplitud clásica.
- `frequency_modulation` (FM): Generación de señales de frecuencia variable.

## Morfología 1D (`morphology.py`)
- Operaciones morfológicas aplicadas a señales unidimensionales para suavizado de envolventes o detección de picos persistentes.
