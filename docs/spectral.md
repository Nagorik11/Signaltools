# Spectral Analysis

Procesamiento de señales en el dominio de la frecuencia.

## Funciones

### `dft(signal)`
Implementación manual de la Transformada Discreta de Fourier. Retorna un diccionario con:
- `real`, `imag`: Componentes complejas.
- `magnitude`: Espectro de amplitud.
- `phase`: Fase de la señal.

### `dominant_bins(magnitude, top_k=8)`
Identifica los bines de frecuencia con mayor energía en el espectro.

### `spectral_energy(magnitude)`
Calcula la energía total del espectro (suma de magnitudes al cuadrado).

### `spectral_flatness(magnitude)`
Mide qué tan tonal o similar al ruido es una señal (Wiener entropy).