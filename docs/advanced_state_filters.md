# signaltools.advanced_state_filters

Documentación compacta del módulo `signaltools.advanced_state_filters`.

Resumen
- Rutinas avanzadas para filtrado no lineal, filtros Bayesianos, suavizado y filtros por partículas.
- Provee resultados tipados mediante dataclasses: `AdaptiveWienerResult`, `NonlinearFilterResult`, `ParticleFilterResult`, `SmootherResult`.

Clases principales
- AdaptiveWienerResult
  - filtered: list[float]
  - noise_trace: list[float]
  - local_variance: list[float]

- NonlinearFilterResult
  - estimates: list[float] | list[list[float]]
  - covariances: list[float] | list[list[list[float]]]
  - meta: dict

- ParticleFilterResult
  - estimates: list[float] | list[list[float]]
  - particles: historial de partículas por paso
  - weights: historial de pesos por paso
  - meta: dict

- SmootherResult
  - estimates, covariances, meta

Funciones principales

1) adaptive_wiener_filter_1d(signal, window_size=5, adaptation_rate=0.1, pad_mode="reflect") -> AdaptiveWienerResult
- Propósito: Filtro de Wiener adaptativo local para señales 1D.
- Parámetros:
  - signal: secuencia 1D (list o array)
  - window_size: impar, tamaño de ventana local
  - adaptation_rate: tasa de actualización del estimador de ruido (>=0)
  - pad_mode: modo de padding para extremos (ej. "reflect")
- Retorna: `AdaptiveWienerResult` con la señal filtrada y trazas de ruido/varianza.
- Errores: ValueError si window_size es par.

Ejemplo:
```python
from signaltools.advanced_state_filters import adaptive_wiener_filter_1d
res = adaptive_wiener_filter_1d([1.0, 0.9, 1.1, 5.0, 1.0], window_size=3)
print(res.filtered)
```

2) extended_kalman_filter(measurements, transition_fn, measurement_fn, transition_jacobian, measurement_jacobian, initial_state, initial_covariance, process_covariance, measurement_covariance) -> NonlinearFilterResult
- Implementa EKF para sistemas no lineales dados f, h y sus jacobianos.
- `transition_fn` y `measurement_fn` reciben y devuelven arrays 1D.
- Devuelve estimaciones por paso y matrices de covarianza.

3) unscented_kalman_filter(measurements, transition_fn, measurement_fn, initial_state, initial_covariance, process_covariance, measurement_covariance, alpha=1e-3, beta=2.0, kappa=0.0) -> NonlinearFilterResult
- Implementa UKF con parámetros de selección de sigma-points.
- `transition_fn` y `measurement_fn` trabajan punto a punto (vector -> vector).

4) particle_filter_1d(measurements, num_particles=100, process_std=0.1, measurement_std=0.2, initial_particles=None, seed=42) -> ParticleFilterResult
- Filtro por partículas sencillo para señal 1D con ruido Gaussiano.
- Retorna estimaciones, trazas de partículas y pesos en cada paso.

Ejemplo mínimo:
```python
from signaltools.advanced_state_filters import particle_filter_1d
meas = [0.0, 0.1, -0.05, 0.0]
res = particle_filter_1d(meas, num_particles=200, process_std=0.05)
print(res.estimates)
```

5) particle_filter_nonlinear(measurements, transition_fn, likelihood_fn, initial_particles, num_particles=None, seed=42) -> ParticleFilterResult
- Filtro por partículas genérico: `transition_fn(particles, rng)` y `likelihood_fn(measurement, particles)` deben ser provistos por el usuario.
- Útil para dinámicas y modelos de observación arbitrarios.

6) particle_filter_multivariate(measurements, transition_fn, likelihood_fn, initial_particles, seed=42) -> ParticleFilterResult
- Extensión multivariante del anterior. `measurements` y `initial_particles` deben ser 2D (secuencia × dimensión).

Suavizadores

- backward_exponential_smoother(signal, alpha=0.3) -> list[float]
  - Suavizado sencillo hacia atrás sobre una secuencia 1D.

- rts_smoother(filtered_estimates, filtered_covariances, transition_matrix, process_covariance) -> SmootherResult
  - Implementación Rauch-Tung-Striebel para suavizar estimaciones de Kalman lineal.
  - `filtered_estimates` debe ser 2D (tiempos × estado), `filtered_covariances` 3D.

Buenas prácticas y notas
- Entradas se convierten a numpy arrays internas; se validan tamaños.
- Para funciones usuario-proveedor (transition_fn, likelihood_fn, measurement_fn) documentar claramente formas de entrada/salida (1D/2D) según la función usada.
- Valores numéricos muy pequeños se protegen internamente con epsilones para evitar divisiones por cero.
- Todas las funciones devuelven dataclasses con método to_dict() para serializar resultados.

Contacto rápido
- Archivo fuente: signaltools/advanced_state_filters.py
- Ubicación recomendada de este documento: src/docs/advanced_state_filters.md

```// filepath: /home/kali/Escritorio/signaltools_src_layout_project/src/docs/advanced_state_filters.md
# signaltools.advanced_state_filters

Documentación compacta del módulo `signaltools.advanced_state_filters`.

Resumen
- Rutinas avanzadas para filtrado no lineal, filtros Bayesianos, suavizado y filtros por partículas.
- Provee resultados tipados mediante dataclasses: `AdaptiveWienerResult`, `NonlinearFilterResult`, `ParticleFilterResult`, `SmootherResult`.

Clases principales
- AdaptiveWienerResult
  - filtered: list[float]
  - noise_trace: list[float]
  - local_variance: list[float]

- NonlinearFilterResult
  - estimates: list[float] | list[list[float]]
  - covariances: list[float] | list[list[list[float]]]
  - meta: dict

- ParticleFilterResult
  - estimates: list[float] | list[list[float]]
  - particles: historial de partículas por paso
  - weights: historial de pesos por paso
  - meta: dict

- SmootherResult
  - estimates, covariances, meta

Funciones principales

1) adaptive_wiener_filter_1d(signal, window_size=5, adaptation_rate=0.1, pad_mode="reflect") -> AdaptiveWienerResult
- Propósito: Filtro de Wiener adaptativo local para señales 1D.
- Parámetros:
  - signal: secuencia 1D (list o array)
  - window_size: impar, tamaño de ventana local
  - adaptation_rate: tasa de actualización del estimador de ruido (>=0)
  - pad_mode: modo de padding para extremos (ej. "reflect")
- Retorna: `AdaptiveWienerResult` con la señal filtrada y trazas de ruido/varianza.
- Errores: ValueError si window_size es par.

Ejemplo:
```python
from signaltools.advanced_state_filters import adaptive_wiener_filter_1d
res = adaptive_wiener_filter_1d([1.0, 0.9, 1.1, 5.0, 1.0], window_size=3)
print(res.filtered)
```

2) extended_kalman_filter(measurements,
