# Módulo Forensics (Criminalística de Señales)

Ubicación: `signaltools/forensics.py`

Este módulo implementa herramientas para garantizar la integridad y trazabilidad de las evidencias digitales durante el análisis de señales.

## Gestión de Evidencias
- `EvidenceManifest`: Contenedor de metadatos de la fuente, incluyendo tamaño, fecha de creación y hashes (MD5, SHA1, SHA256).
- `create_evidence_manifest()`: Genera automáticamente el manifiesto para archivos, arrays de NumPy o listas.

## Integridad y Seguridad
- `hash_file()` / `hash_bytes()`: Generan múltiples sumas de verificación para asegurar que la evidencia no ha sido alterada.
- `TimestampSeal`: Sello de tiempo digital que vincula una evidencia con un momento específico.
- `ReportSignature`: Firma digital del reporte final (soporta HMAC-SHA256 y SHA256-DIGEST).

## Cadena de Custodia
- `ChainOfCustody`: Registro cronológico de quién, qué, cuándo y dónde ha tenido acceso o ha modificado la evidencia.
- `append_chain_of_custody_event()`: Añade un nuevo evento (adquisición, análisis, sellado) al historial.

## Análisis Pericial Completo
- `forensic_analyze_signal()`: Función de alto nivel que:
    1. Carga la señal.
    2. Realiza un análisis avanzado.
    3. Registra todos los pasos en una pista de auditoría (`AuditStep`).
    4. Genera la cadena de custodia inicial.
    5. Firma el reporte final.
- `write_forensic_bundle()`: Exporta todos los componentes (manifiesto, análisis, auditoría, custodia) en un paquete de archivos JSON organizado dentro de un directorio.
