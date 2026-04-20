# Módulo Wavelets (Paquetes de Ondículas)

Este módulo proporciona una implementación extensa de descomposiciones de paquetes de wavelets en múltiples dimensiones (1D a 5D).

## Descomposición 1D
Ubicación: `signaltools/wavelet_packet.py`

- `WaveletPacketTree`: Estructura que contiene los nodos del árbol de descomposición.
- `wavelet_packet_decompose(signal, level, family)`: Realiza la descomposición completa.
- `wavelet_packet_reconstruct(tree)`: Reconstruye la señal original a partir del árbol.

## Descomposiciones Multidimensionales
- `wavelet_packet_2d_decompose`: Para imágenes.
- `wavelet_packet_3d_decompose`: Para volúmenes de datos.
- `wavelet_packet_4d_decompose`: Para datos 4D.
- `wavelet_packet_5d_decompose`: Para tensores de 5 dimensiones (p. ej., hiperespectrales).

## Familias de Wavelets Soportadas
El sistema soporta familias ortogonales y biortogonales, incluyendo:
- `Haar`
- `Daubechies` (db2, db4, db6, db8)
- `Symlets` (sym2, sym4, sym6)
- `Coiflets` (coif1, coif2, coif3)
- `Biorthogonal` (bior53, bior97, etc.)

## Capacidades Adaptativas (5D)
Ubicación: `signaltools/wavelet_packet_5d.py`

- `adaptive_wavelet_packet_5d_decompose`: Selecciona automáticamente la mejor familia para cada eje.
- `spatially_variable_wavelet_packet_5d_decompose`: Adapta la familia de forma independiente en cada nodo del árbol.
- `subband_attentive_wavelet_packet_5d_decompose`: Utiliza mecanismos de atención para mezclar sub-bandas durante la descomposición.
