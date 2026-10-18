# -*- coding: utf-8 -*-
"""
App de demostración del RETAINIQ-AGENT, lista para desplegar en
Streamlit Community Cloud (streamlit.io/cloud).

Para desplegarla:
  1. Suban este repositorio completo a GitHub (público o con acceso
     para el docente).
  2. Entren a https://share.streamlit.io con su cuenta de GitHub.
  3. "New app" -> seleccionen el repositorio -> main file: app.py.
  4. Esperen el build (2-3 minutos) y obtendrán un enlace público.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from agent import RetainIQAgent
from entorno import EntornoRRHH

st.set_page_config(page_title="RETAINIQ-AGENT", page_icon="🧭", layout="wide")

st.title("🧭 RETAINIQ-AGENT")
st.caption(
    "Agente inteligente de predicción de rotación de empleados — "
    "Universidad Tecnológica de Bolívar, curso de Inteligencia Artificial. "
    "Implementación de David Peláez Mejía y Jose Pereira Acuña."
)


@st.cache_resource
def cargar_agente():
    entorno = EntornoRRHH("data/HR-Employee-Attrition.csv", seed=1)
    X, y = entorno.observar()
    agente = RetainIQAgent(log_dir="logs_app")
    metricas = agente.entrenar(X, y, seed=1)
    return agente, entorno, X, y, metricas


agente, entorno, X_base, y_base, metricas_entrenamiento = cargar_agente()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dashboard mensual", "🔎 Evaluar un empleado", "⚠️ Escenario de falla", "📈 Métricas del modelo"]
)

# --------------------------------------------------------------------
with tab1:
    st.subheader("Ciclo mensual simulado")
    n = st.slider("Número de empleados a procesar en este ciclo", 10, len(X_base), 200)
    if st.button("▶️ Ejecutar ciclo mensual", type="primary"):
        with st.spinner("Percepción → Decisión → Acción → Comunicación → Memoria..."):
            resultado = agente.ejecutar_ciclo_mensual(X_base.head(n), ciclo_id="demo")
        resumen = resultado["resumen"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total evaluados", resumen["total_empleados"])
        c2.metric("Riesgo alto", resumen["riesgo_alto"])
        c3.metric("Riesgo medio", resumen["riesgo_medio"])
        c4.metric("Probabilidad promedio", f"{resumen['probabilidad_promedio']:.1%}")

        st.markdown("**Reporte de riesgo (salida del agente — Sección 6 del diseño):**")
        st.dataframe(resultado["decisiones"].sort_values("probabilidad_renuncia", ascending=False),
                     use_container_width=True)

        fig, ax = plt.subplots()
        resultado["decisiones"]["categoria_riesgo"].value_counts().reindex(
            ["alto", "medio", "bajo"]
        ).plot(kind="bar", ax=ax, color=["#c0392b", "#f39c12", "#27ae60"])
        ax.set_ylabel("Número de empleados")
        ax.set_title("Distribución de riesgo del ciclo")
        st.pyplot(fig)

# --------------------------------------------------------------------
with tab2:
    st.subheader("Evaluar un empleado individual")
    idx = st.selectbox("Selecciona un EmployeeNumber", X_base["EmployeeNumber"].tolist())
    fila = X_base[X_base["EmployeeNumber"] == idx]
    st.dataframe(fila, use_container_width=True)
    if st.button("Evaluar riesgo de este empleado"):
        pred = agente.decidir(fila)
        fila_pred = pred.iloc[0]
        color = {"alto": "🔴", "medio": "🟡", "bajo": "🟢"}[fila_pred["categoria_riesgo"]]
        st.markdown(
            f"### {color} Riesgo **{fila_pred['categoria_riesgo'].upper()}** "
            f"— probabilidad de renuncia: **{fila_pred['probabilidad_renuncia']:.1%}**"
        )

# --------------------------------------------------------------------
with tab3:
    st.subheader("Escenario de falla esperada: sensor sin dato")
    st.write(
        "Simula que el HRIS o el sistema de evaluación no entregó algunas "
        "variables (valores faltantes). El agente debe imputar y continuar "
        "sin lanzar una excepción."
    )
    if st.button("Simular falla de sensor"):
        dañado = entorno.inyectar_fallo_sensor(
            X_base.head(30), columnas=["JobSatisfaction", "MonthlyIncome", "EnvironmentSatisfaction"],
            fraccion=0.3,
        )
        try:
            pred = agente.decidir(dañado)
            st.success(f"✅ El agente procesó {len(pred)} registros sin error pese a los datos faltantes.")
            st.dataframe(pred, use_container_width=True)
        except Exception as e:
            st.error(f"❌ El agente falló: {e}")

# --------------------------------------------------------------------
with tab4:
    st.subheader("Métricas del modelo (entrenamiento inicial)")
    st.json(metricas_entrenamiento)
    st.caption(
        "Medida de desempeño SMART (Sección 7 del diseño): precisión ≥ 75% "
        "en la clasificación de empleados de alto riesgo. Ver el análisis "
        "completo de la Fase 3 en logs/resultados_fase3.csv del repositorio."
    )
