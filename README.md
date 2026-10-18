# RETAINIQ-AGENT

Implementación del agente inteligente de predicción de rotación de empleados, desarrollado para la actividad "Implementación del Agente Diseñado" del curso de Inteligencia Artificial (Universidad Tecnológica de Bolívar).

**Equipo:** David Peláez Mejía (T00079273) y Jose Pereira Acuña (T00079768).

**Documento de diseño base:** `Diseño_Agente_RetainIQ_Ajustado.pdf` (Fase 1 y Fase 2 adjuntas a la entrega).

## 1. Qué implementa este repositorio

Un agente **basado en utilidad con capacidades de aprendizaje** (clasificación AIMA, Sección 2 del diseño) que:

1. **Percibe** datos de empleados (HRIS, nómina, evaluación de desempeño simulados).
2. **Decide** una probabilidad de renuncia y una categoría de riesgo (alto/medio/bajo) con un modelo Random Forest.
3. **Actúa** registrando un reporte y notificando a los líderes de los casos de alto riesgo.
4. **Se comunica** mediante un resumen estructurado del ciclo.
5. **Recuerda** el histórico de ciclos y resultados reales (memoria de corto y largo plazo).
6. **Aprende**, recalibrando el modelo con los resultados reales observados.

El ciclo completo (percepción → decisión → acción) es explícito en `agent.py`, método `ejecutar_ciclo_mensual()`.

## 2. Estructura del repositorio

```
retainiq_agent/
├── agent.py                  # Núcleo del agente (7 componentes IBM, métodos documentados)
├── entorno.py                 # Simulador del entorno (estocástico, dinámico, secuencial)
├── memoria.py                 # Memoria de corto y largo plazo
├── app.py                     # App de demostración (Streamlit)
├── diagrama_secuencia.mmd     # Diagrama de secuencia UML (Mermaid) — se renderiza solo en GitHub
├── diagrama_arquitectura.png  # Diagrama de componentes
├── requirements.txt
├── data/
│   └── HR-Employee-Attrition.csv   # Dataset IBM HR Analytics (1,470 registros, 35 variables)
├── logs/
│   ├── log_ejecucion.json          # Log de ejecución con marca de tiempo
│   ├── memoria_largo_plazo.json    # Histórico de ciclos
│   ├── reporte_ciclo_*.csv         # Reportes generados por el agente
│   ├── notificacion_ciclo_*.json   # Notificaciones simuladas
│   ├── resultados_fase3.csv        # Tabla de resultados de las pruebas
│   └── resultados_fase3.json       # Detalle de las pruebas (por semilla)
└── tests/
    └── test_fase3.py          # Arnés de pruebas de la Fase 3
```

## 3. Cómo ejecutarlo

```bash
git clone <url-de-este-repositorio>
cd retainiq_agent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Ejecutar las pruebas de la Fase 3 (genera logs/resultados_fase3.csv)
python3 tests/test_fase3.py

# Levantar la demo interactiva
streamlit run app.py
```

## 4. Diagrama de secuencia

El archivo `diagrama_secuencia.mmd` contiene el diagrama UML en formato Mermaid. GitHub lo renderiza automáticamente si lo pegas dentro de un bloque ```mermaid en este README, o puedes verlo en [mermaid.live](https://mermaid.live) pegando su contenido.

## 5. Diferencias entre diseño e implementación

Ver la sección 9.5 ("Diferencias entre diseño e implementación") del documento de diseño ajustado — resumen: el HRIS, el sistema de nómina y el sistema de evaluación reales se sustituyeron por un simulador (`entorno.py`) que reproduce sus propiedades de ambiente (parcialmente observable, estocástico, dinámico, secuencial) a partir del dataset IBM HR Analytics; SHAP (mencionado en el diseño original) no se implementó por restricciones de tiempo — se documenta como trabajo futuro.

## 6. Bitácora de uso de IA

Ver `BITACORA_IA.md` en la raíz de la entrega.
