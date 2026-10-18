# -*- coding: utf-8 -*-
"""
Fase 3 — Pruebas basadas en las medidas de desempeño.

Convierte cada medida SMART de la Sección 7 del diseño en pruebas con
resultado numérico, usando >= 10 escenarios (normales, límite y de
falla esperada) y, donde hay aleatoriedad, >= 5 repeticiones con media
y desviación estándar, tal como exige la Fase 3 de la actividad.
"""
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import RetainIQAgent
from entorno import EntornoRRHH
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

RUTA_DATOS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "HR-Employee-Attrition.csv")
RUTA_RESULTADOS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_fase3.csv")
RUTA_RESULTADOS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_fase3.json")

UMBRAL_PRECISION = 0.75
UMBRAL_F1 = 0.50  # umbral complementario propuesto para el análisis


def escenario_normal(seed, fraccion_test=0.2):
    """Escenario normal: entrenamiento/evaluación con una semilla fija,
    dataset completo sin alteraciones."""
    entorno = EntornoRRHH(RUTA_DATOS, seed=seed)
    X, y = entorno.observar()
    agente = RetainIQAgent(log_dir=f"/tmp/retainiq_test_logs/normal_{seed}")
    m_train = agente.entrenar(X, y, test_size=fraccion_test, seed=seed)
    return m_train


def escenario_limite_empleado_extremo(seed):
    """Caso límite: empleado con antigüedad y edad en los extremos del
    rango observado en el dataset."""
    entorno = EntornoRRHH(RUTA_DATOS, seed=seed)
    X, y = entorno.observar()
    agente = RetainIQAgent(log_dir=f"/tmp/retainiq_test_logs/limite_{seed}")
    agente.entrenar(X, y, seed=seed)
    extremo = X.loc[[X["TotalWorkingYears"].idxmax()]].copy()
    extremo["EmployeeNumber"] = 999001
    pred = agente.decidir(extremo)
    return pred.iloc[0].to_dict()


def escenario_limite_empleado_nuevo(seed):
    """Caso límite: empleado recién contratado (antigüedad y datos
    mínimos)."""
    entorno = EntornoRRHH(RUTA_DATOS, seed=seed)
    X, y = entorno.observar()
    agente = RetainIQAgent(log_dir=f"/tmp/retainiq_test_logs/nuevo_{seed}")
    agente.entrenar(X, y, seed=seed)
    nuevo = X.iloc[[0]].copy()
    nuevo["YearsAtCompany"] = 0
    nuevo["TotalWorkingYears"] = 0
    nuevo["NumCompaniesWorked"] = 0
    nuevo["EmployeeNumber"] = 999002
    pred = agente.decidir(nuevo)
    return pred.iloc[0].to_dict()


def escenario_falla_sensor_sin_dato(seed):
    """Falla esperada: variables clave llegan sin dato (NaN), simulando
    un sensor/API caído. El agente no debe lanzar una excepción."""
    entorno = EntornoRRHH(RUTA_DATOS, seed=seed)
    X, y = entorno.observar()
    agente = RetainIQAgent(log_dir=f"/tmp/retainiq_test_logs/fallo_sensor_{seed}")
    agente.entrenar(X, y, seed=seed)
    X_dañado = entorno.inyectar_fallo_sensor(
        X.head(30), columnas=["JobSatisfaction", "MonthlyIncome", "EnvironmentSatisfaction"], fraccion=0.3
    )
    try:
        pred = agente.decidir(X_dañado)
        return {"exito": True, "n_procesados": len(pred), "error": None}
    except Exception as e:
        return {"exito": False, "n_procesados": 0, "error": str(e)}


def escenario_falla_entrada_invalida(seed):
    """Falla esperada: lote de entrada vacío."""
    entorno = EntornoRRHH(RUTA_DATOS, seed=seed)
    X, y = entorno.observar()
    agente = RetainIQAgent(log_dir=f"/tmp/retainiq_test_logs/fallo_vacio_{seed}")
    agente.entrenar(X, y, seed=seed)
    vacio = X.iloc[0:0]
    try:
        pred = agente.decidir(vacio)
        return {"exito": True, "n_procesados": len(pred), "error": None}
    except Exception as e:
        return {"exito": False, "n_procesados": 0, "error": str(e)}


def escenario_reestructuracion(seed):
    """Condición excepcional (Sección 4): reestructuración
    organizacional con recorte salarial simulado; se espera que el
    riesgo promedio aumente frente al ciclo normal."""
    entorno = EntornoRRHH(RUTA_DATOS, seed=seed)
    X, y = entorno.observar()
    agente = RetainIQAgent(log_dir=f"/tmp/retainiq_test_logs/reestructuracion_{seed}")
    agente.entrenar(X, y, seed=seed)
    pred_normal = agente.decidir(X.head(100))
    X_reestructurado = entorno.reestructuracion_organizacional(X.head(100))
    pred_reestructurado = agente.decidir(X_reestructurado)
    return {
        "prob_promedio_normal": round(float(pred_normal["probabilidad_renuncia"].mean()), 4),
        "prob_promedio_reestructuracion": round(float(pred_reestructurado["probabilidad_renuncia"].mean()), 4),
        "aumento_riesgo": bool(
            pred_reestructurado["probabilidad_renuncia"].mean() > pred_normal["probabilidad_renuncia"].mean()
        ),
    }


def ejecutar_fase3():
    resultados = []

    # --- Medida 1: Precisión >= 75% (5 repeticiones con distinta semilla) ---
    precisiones, recalls, f1s, aucs = [], [], [], []
    for seed in [1, 2, 3, 4, 5]:
        m = escenario_normal(seed)
        precisiones.append(m["precision"]); recalls.append(m["recall"])
        f1s.append(m["f1"]); aucs.append(m["auc_roc"])

    resultados.append({
        "medida": "Precisión en clasificación de riesgo alto",
        "formula": "precision_score(y_true, y_pred) con umbral de decisión 0.5",
        "umbral": f">= {UMBRAL_PRECISION:.0%}",
        "casos_ejecutados": 5,
        "media": round(float(np.mean(precisiones)), 4),
        "desviacion_estandar": round(float(np.std(precisiones)), 4),
        "cumple": bool(np.mean(precisiones) >= UMBRAL_PRECISION),
        "evidencia": "logs/resultados_fase3.json -> precision_por_seed",
    })
    resultados.append({
        "medida": "Recall (sensibilidad) complementaria",
        "formula": "recall_score(y_true, y_pred)",
        "umbral": "informativo (sin umbral SMART propio)",
        "casos_ejecutados": 5,
        "media": round(float(np.mean(recalls)), 4),
        "desviacion_estandar": round(float(np.std(recalls)), 4),
        "cumple": None,
        "evidencia": "logs/resultados_fase3.json -> recall_por_seed",
    })
    resultados.append({
        "medida": "F1-Score complementario",
        "formula": "f1_score(y_true, y_pred)",
        "umbral": f">= {UMBRAL_F1:.0%} (propuesto para este análisis)",
        "casos_ejecutados": 5,
        "media": round(float(np.mean(f1s)), 4),
        "desviacion_estandar": round(float(np.std(f1s)), 4),
        "cumple": bool(np.mean(f1s) >= UMBRAL_F1),
        "evidencia": "logs/resultados_fase3.json -> f1_por_seed",
    })
    resultados.append({
        "medida": "AUC-ROC complementaria",
        "formula": "roc_auc_score(y_true, probas)",
        "umbral": ">= 0.70 (propuesto para este análisis)",
        "casos_ejecutados": 5,
        "media": round(float(np.mean(aucs)), 4),
        "desviacion_estandar": round(float(np.std(aucs)), 4),
        "cumple": bool(np.mean(aucs) >= 0.70),
        "evidencia": "logs/resultados_fase3.json -> auc_por_seed",
    })

    # --- Casos límite ---
    lim1 = escenario_limite_empleado_extremo(seed=1)
    resultados.append({
        "medida": "Caso límite: empleado con antigüedad máxima",
        "formula": "Clasificación individual (decidir())",
        "umbral": "El agente debe producir una categoría válida sin error",
        "casos_ejecutados": 1,
        "media": lim1["probabilidad_renuncia"],
        "desviacion_estandar": None,
        "cumple": lim1["categoria_riesgo"] in ["alto", "medio", "bajo"],
        "evidencia": f"categoria_riesgo={lim1['categoria_riesgo']}",
    })
    lim2 = escenario_limite_empleado_nuevo(seed=1)
    resultados.append({
        "medida": "Caso límite: empleado recién contratado",
        "formula": "Clasificación individual (decidir())",
        "umbral": "El agente debe producir una categoría válida sin error",
        "casos_ejecutados": 1,
        "media": lim2["probabilidad_renuncia"],
        "desviacion_estandar": None,
        "cumple": lim2["categoria_riesgo"] in ["alto", "medio", "bajo"],
        "evidencia": f"categoria_riesgo={lim2['categoria_riesgo']}",
    })

    # --- Casos de falla esperada (mínimo 2, requerido por la guía) ---
    f1_fail = escenario_falla_sensor_sin_dato(seed=1)
    resultados.append({
        "medida": "Falla esperada: sensor sin dato (30% NaN en 3 variables)",
        "formula": "El agente debe imputar y continuar sin excepción",
        "umbral": "exito == True",
        "casos_ejecutados": 1,
        "media": None, "desviacion_estandar": None,
        "cumple": f1_fail["exito"],
        "evidencia": f"n_procesados={f1_fail['n_procesados']}, error={f1_fail['error']}",
    })
    f2_fail = escenario_falla_entrada_invalida(seed=1)
    resultados.append({
        "medida": "Falla esperada: lote de entrada vacío",
        "formula": "El agente debe manejar el caso sin excepción",
        "umbral": "exito == True",
        "casos_ejecutados": 1,
        "media": None, "desviacion_estandar": None,
        "cumple": f2_fail["exito"],
        "evidencia": f"n_procesados={f2_fail['n_procesados']}, error={f2_fail['error']}",
    })

    # --- Condición excepcional del entorno ---
    reest = escenario_reestructuracion(seed=1)
    resultados.append({
        "medida": "Condición excepcional: reestructuración organizacional",
        "formula": "prob_promedio(post) > prob_promedio(pre)",
        "umbral": "El riesgo promedio debe aumentar tras el recorte salarial simulado",
        "casos_ejecutados": 100,
        "media": reest["prob_promedio_reestructuracion"],
        "desviacion_estandar": None,
        "cumple": reest["aumento_riesgo"],
        "evidencia": f"antes={reest['prob_promedio_normal']}, despues={reest['prob_promedio_reestructuracion']}",
    })

    # --- Reducción de tiempo de procesamiento manual (estimación) ---
    # Línea base estimada: revisión manual de RR.HH. ~ 8 min/empleado (criterio experto)
    # Tiempo del agente: medido por ejecución real sobre 100 empleados
    import time
    entorno = EntornoRRHH(RUTA_DATOS, seed=1)
    X, y = entorno.observar()
    agente = RetainIQAgent(log_dir="/tmp/retainiq_test_logs/tiempo")
    agente.entrenar(X, y, seed=1)
    t0 = time.time()
    agente.decidir(X.head(100))
    t_agente = time.time() - t0
    t_manual_estimado = 100 * 8 * 60  # 8 min/empleado en segundos
    reduccion = 1 - (t_agente / t_manual_estimado)
    resultados.append({
        "medida": "Reducción de tiempo de procesamiento manual",
        "formula": "1 - (tiempo_agente / tiempo_manual_estimado)",
        "umbral": "Informativo — línea base estimada, no medida en campo real",
        "casos_ejecutados": 100,
        "media": round(reduccion, 4),
        "desviacion_estandar": None,
        "cumple": None,
        "evidencia": f"tiempo_agente={t_agente:.4f}s para 100 empleados vs. línea base estimada de {t_manual_estimado}s",
    })

    # Guardar resultados
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(RUTA_RESULTADOS_CSV, index=False)
    detalle_json = {
        "precision_por_seed": dict(zip([1, 2, 3, 4, 5], [round(p, 4) for p in precisiones])),
        "recall_por_seed": dict(zip([1, 2, 3, 4, 5], [round(r, 4) for r in recalls])),
        "f1_por_seed": dict(zip([1, 2, 3, 4, 5], [round(f, 4) for f in f1s])),
        "auc_por_seed": dict(zip([1, 2, 3, 4, 5], [round(a, 4) for a in aucs])),
        "resultados_tabla": resultados,
    }
    with open(RUTA_RESULTADOS_JSON, "w", encoding="utf-8") as f:
        json.dump(detalle_json, f, ensure_ascii=False, indent=2, default=str)

    return df_resultados


if __name__ == "__main__":
    df = ejecutar_fase3()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))
    print(f"\nResultados guardados en:\n  {RUTA_RESULTADOS_CSV}\n  {RUTA_RESULTADOS_JSON}")
