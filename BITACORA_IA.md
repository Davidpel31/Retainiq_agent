# Bitácora de uso de IA

**Equipo:** David Peláez Mejía (T00079273) y Jose Pereira Acuña (T00079768)
**Actividad:** Implementación del Agente Diseñado — RETAINIQ-AGENT
**Herramienta:** Claude (Anthropic), usado desde la cuenta de David Peláez Mejía
**Fecha:** 18 de octubre de 2026

## 1. Resumen

Claude se usó como asistente en las tres fases de la actividad. Generó la propuesta inicial de la matriz de decisión, el código del agente, las pruebas, los diagramas, el README y la redacción de la nueva sección de desarrollo del documento. El equipo revisó ese material, ejecutó el código en su propio equipo, tomó la decisión final de qué agente implementar y corrigió lo que no coincidía con lo observado.

## 2. Qué hizo la IA y qué hizo el equipo, por fase

### Fase 1 — Selección del agente

**IA.** Los seis criterios y sus pesos sugeridos vienen de la guía de la actividad. Claude propuso los puntajes de 1 a 5 y la evidencia de cada celda para InvoiceFlow-Agent y RetainIQ-Agent, calculó el puntaje ponderado (3.65 y 4.90) y ejecutó el análisis de sensibilidad (±10 puntos sobre el criterio de mayor peso).

**Equipo.** Ambos integrantes revisaron la matriz. La decisión de implementar RetainIQ-Agent la tomó el equipo con su propio criterio, que coincide con el resultado de la matriz pero no se limita a ella. Los dos concluyeron que el agente de Jose era más viable por estas razones:

- No depende de terceros: usa librerías de código abierto y un dataset público ya estructurado. InvoiceFlow-Agent depende de un motor de reconocimiento óptico de caracteres y de integraciones con un ERP y con órdenes de compra.
- La extracción de texto de facturas está expuesta a la calidad de los datos de entrada (escaneo, formatos, errores de reconocimiento).
- El manejo de datos de InvoiceFlow-Agent es más complejo: documentos financieros de terceros y datos que habría que simular.
- Construir la aplicación de InvoiceFlow-Agent era más complicado en general.

Esas razones quedaron escritas en la justificación de la elección del documento.

### Fase 2 — Desarrollo

**IA.** Claude generó el código de `agent.py`, `entorno.py`, `memoria.py` y `app.py`, los diagramas de arquitectura y de secuencia, y el README. El dataset (IBM HR Analytics) lo descargó desde una copia pública en GitHub.

**Equipo.** David descargó el código, lo ejecutó en Windows, levantó la aplicación en local, creó el repositorio en GitHub, fijó las versiones exactas en `requirements.txt` y sincronizó el repositorio. Ambos integrantes revisaron el código.

### Fase 3 — Pruebas

**IA.** Claude escribió el arnés de pruebas (`tests/test_fase3.py`), primero con 10 filas de resultados y luego, en una segunda versión, con 10 repeticiones para la medida principal, dos escenarios adversos adicionales y 5 repeticiones en los demás casos.

**Equipo.** David ejecutó las pruebas en su máquina y los resultados del documento corresponden a esa ejecución (Python 3.12.10, scikit-learn 1.9.1, pandas 3.0.6, NumPy 2.5.3). La medida principal no alcanzó la meta: precisión de 47.7 % frente al 75 % del diseño. El equipo decidió reportarlo tal como salió, sin modificar el modelo ni los umbrales para forzar el cumplimiento.

Una ejecución previa de validación hecha por Claude, con otra versión de scikit-learn, dio números distintos. Esa diferencia está explicada en el documento (Sección 9.7).

## 3. Trabajo propio del equipo

Además de revisar lo que generó la IA, el equipo hizo lo siguiente por su cuenta:

- Comparó los dos diseños de la entrega anterior y evaluó cuál era más viable de implementar, decisión que tomaron los dos.
- Preparó el entorno de ejecución en Windows y resolvió los problemas que aparecieron (uso de `python` en lugar de `python3`, activación del entorno virtual, instalación de dependencias).
- Ejecutó el agente, la aplicación y las pruebas, y comparó los resultados con los del documento. Al ver que los números cambiaban según la versión de las librerías, fijó las versiones exactas en `requirements.txt`.
- Creó y mantuvo el repositorio en GitHub, y resolvió la diferencia entre la copia local y la remota.
- Decidió reportar los resultados tal como salieron, incluido el incumplimiento de la meta de precisión.

## 4. Documento

- Secciones 1 a 8: corresponden al diseño original de Jose Pereira Acuña (RetainIQ-Agent), reformateado en la plantilla del curso sin cambiar su contenido técnico.
- Matriz de decisión, tabla de cambios y Sección 9 (desarrollo del agente): redactadas con asistencia de Claude y revisadas por el equipo. Las cifras de la Sección 9.7 vienen de la ejecución de David.

## 5. Límites del uso de IA

- La IA no tuvo acceso a la cuenta de GitHub ni al equipo del estudiante; todo lo que está en el repositorio lo subió David.
- La línea base de tiempo de procesamiento manual (8 minutos por empleado) es una estimación razonada, no una medición de campo.
- La IA no implementó SHAP ni la detección automática de sesgos que mencionaba el diseño; ambos quedan declarados como trabajo futuro.

## 6. Declaración

El equipo declara que revisó el código, el documento y esta bitácora, que comprende el funcionamiento del agente y que asume la responsabilidad académica sobre el contenido entregado.