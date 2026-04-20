# Estimación de Estado y Filtrado Avanzado (Detalle Técnico)

Este documento describe los filtros de estado utilizados para seguimiento, estimación y eliminación de ruido en señales dinámicas.

## Filtros de Kalman (`state_filters.py` / `advanced_state_filters.py`)
- `KalmanFilter1D`: Filtro de Kalman estándar para sistemas lineales.
- `ExtendedKalmanFilter` (EKF): Soporte para transiciones y mediciones no lineales mediante linealización (Jacobianos).
- `UnscentedKalmanFilter` (UKF): Utiliza la transformada unscented para capturar mejor la media y covarianza en sistemas altamente no lineales.

## Filtrado de Partículas (`advanced_state_filters.py`)
- `ParticleFilter1D`: Estimación de estado mediante Monte Carlo secuencial, ideal para distribuciones no Gaussianas y sistemas arbitrarios.

## Filtros Óptimos y Adaptativos
- `AdaptiveWienerFilter`: Minimización del error cuadrático medio mediante estimación local de varianza.
- `LMSAdaptiveFilter`: Filtro Least Mean Squares para cancelación de ruido e identificación de sistemas.

## Bancos de Filtros (`filter_banks.py`)
Implementación de estructuras para la división y reconstrucción de señales en múltiples sub-bandas.
