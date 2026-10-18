# -*- coding: utf-8 -*-
"""
Fase 3 — Pruebas basadas en las medidas de desempeño (versión 2).

Cambios frente a la versión anterior:
  - La medida principal y sus métricas complementarias se evalúan con 10 semillas.
  - La medida principal se evalúa también bajo dos condiciones adversas
    (sensor sin dato y reestructuración organizacional), con 10 semillas cada una.
  - Los casos límite, de falla esperada y de condición excepcional se repiten con
    5 semillas y se reporta media y desviación estándar.
  - Los resultados se guardan en tests/ y en logs/.

Los umbrales de aceptación están definidos como constantes en este archivo,
antes de ejecutar cualquier prueba.
"""
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from agent import RetainIQAgent  # noqa: E402
from entorno import EntornoRRHH  # noqa: E402

RUTA_DATOS = os.path.join(RAIZ, "data", "HR-Employee-Attrition.csv")
DIR_TESTS = os.path.dirname(os.path.abspath(__file__))
DIR_LOGS = os.path.join(RAIZ, "logs")
TMP = os.path.join(tempfile.gettempdir(), "retainiq_test_logs")

# ---- Umbrales de aceptación (fijados antes de ejecutar las pruebas) ----------
UMBRAL_PRECISION = 0.75   # meta SMART del diseño (Sección 7)
UMBRAL_F1 = 0.50          # propuesto para este análisis
UMBRAL_AUC = 0.70         # propuesto para este análisis
UMBRAL_AUC_ADVERSO = 0.70 # el modelo debe conservar capacidad de ordenar casos bajo condiciones adversas
SEMILLAS_PRINCIPAL = list(range(1, 11))   # 10 repeticiones
SEMILLAS_ESCENARIOS = list(range(1, 6))   # 5 repeticiones
MINUTOS_MANUAL_POR_EMPLEADO = 8           # línea base estimada (no medida en campo)


def probas(agente, df_crudo):
    X = agente.percibir(df_crudo).reindex(columns=agente.features, fill_value=0)
    return agente.modelo.predict_proba(X)[:, 1]


def metricas(y_true_bin, p, umbral=0.5):
    y_pred = (p >= umbral).astype(int)
    return {
        "precision": float(precision_score(y_true_bin, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_bin, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_bin, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true_bin, p)),
    }


def categoria_esperada(prob):
    if prob >= RetainIQAgent.UMBRAL_ALTO:
        return "alto"
    if prob >= RetainIQAgent.UMBRAL_MEDIO:
        return "medio"
    return "bajo"


def ms(valores):
    return round(float(np.mean(valores)), 4), round(float(np.std(valores)), 4)


def ejecutar_fase3():
    normal = {k: [] for k in ["precision", "recall", "f1", "auc_roc"]}
    sensor = {k: [] for k in ["precision", "recall", "f1", "auc_roc"]}
    reest = {k: [] for k in ["precision", "recall", "f1", "auc_roc"]}
    lim_extremo, lim_nuevo = [], []        # (prob, categoria_valida)
    falla_sensor, falla_vacio = [], []     # exito
    exc_antes, exc_despues, exc_aumento = [], [], []
    tiempos = []

    for seed in SEMILLAS_PRINCIPAL:
        entorno = EntornoRRHH(RUTA_DATOS, seed=seed)
        X, y = entorno.observar()
        agente = RetainIQAgent(log_dir=os.path.join(TMP, f"seed_{seed}"))
        m_train = agente.entrenar(X, y, seed=seed)

        # --- Medida principal (escenario normal): métricas del propio entrenamiento ---
        for k in normal:
            normal[k].append(m_train[k if k != "auc_roc" else "auc_roc"])

        # --- Conjunto de prueba reproducido (mismo split que usa agent.entrenar) ---
        y_bin = (y == "Yes").astype(int).values
        _, idx_test = train_test_split(
            np.arange(len(y_bin)), test_size=0.2, random_state=seed, stratify=y_bin
        )
        X_test, yb_test = X.iloc[idx_test], y_bin[idx_test]

        # Verificación de consistencia: el evaluador debe reproducir las métricas del entrenamiento
        m_check = metricas(yb_test, probas(agente, X_test))
        if abs(m_check["precision"] - m_train["precision"]) > 1e-3:
            raise RuntimeError(f"Inconsistencia de evaluación en semilla {seed}")

        # --- Escenario adverso 1: sensor sin dato (30 % NaN en 3 variables) ---
        X_s = entorno.inyectar_fallo_sensor(
            X_test, ["JobSatisfaction", "MonthlyIncome", "EnvironmentSatisfaction"], fraccion=0.3
        )
        ms_ = metricas(yb_test, probas(agente, X_s))
        for k in sensor:
            sensor[k].append(ms_[k])

        # --- Escenario adverso 2: reestructuración organizacional ---
        X_r = entorno.reestructuracion_organizacional(X_test)
        mr_ = metricas(yb_test, probas(agente, X_r))
        for k in reest:
            reest[k].append(mr_[k])

        # --- Escenarios con 5 repeticiones ---
        if seed in SEMILLAS_ESCENARIOS:
            extremo = X.loc[[X["TotalWorkingYears"].idxmax()]].copy()
            extremo["EmployeeNumber"] = 999001
            r = agente.decidir(extremo).iloc[0]
            lim_extremo.append((float(r["probabilidad_renuncia"]),
                                0 <= r["probabilidad_renuncia"] <= 1 and
                                r["categoria_riesgo"] == categoria_esperada(r["probabilidad_renuncia"])))

            nuevo = X.iloc[[0]].copy()
            nuevo["YearsAtCompany"] = 0
            nuevo["TotalWorkingYears"] = 0
            nuevo["NumCompaniesWorked"] = 0
            nuevo["EmployeeNumber"] = 999002
            r = agente.decidir(nuevo).iloc[0]
            lim_nuevo.append((float(r["probabilidad_renuncia"]),
                              0 <= r["probabilidad_renuncia"] <= 1 and
                              r["categoria_riesgo"] == categoria_esperada(r["probabilidad_renuncia"])))

            try:
                dañado = entorno.inyectar_fallo_sensor(
                    X.head(30), ["JobSatisfaction", "MonthlyIncome", "EnvironmentSatisfaction"], 0.3
                )
                falla_sensor.append(len(agente.decidir(dañado)) == 30)
            except Exception:
                falla_sensor.append(False)
            try:
                falla_vacio.append(len(agente.decidir(X.iloc[0:0])) == 0)
            except Exception:
                falla_vacio.append(False)

            antes = agente.decidir(X.head(100))["probabilidad_renuncia"].mean()
            despues = agente.decidir(
                entorno.reestructuracion_organizacional(X.head(100))
            )["probabilidad_renuncia"].mean()
            exc_antes.append(float(antes))
            exc_despues.append(float(despues))
            exc_aumento.append(bool(despues > antes))

            t0 = time.perf_counter()
            agente.decidir(X.head(100))
            tiempos.append(time.perf_counter() - t0)

    # ------------------------------------------------------------------
    filas = []

    def fila(medida, formula, umbral, n, media, desv, cumple, evidencia):
        filas.append({
            "medida": medida, "formula": formula, "umbral": umbral,
            "casos_ejecutados": n, "media": media, "desviacion_estandar": desv,
            "cumple": cumple, "evidencia": evidencia,
        })

    n_p = len(SEMILLAS_PRINCIPAL)
    m, s = ms(normal["precision"])
    fila("Precisión en clasificación de riesgo alto (escenario normal)",
         "precision_score(y_true, y_pred), umbral de decisión 0.5", f">= {UMBRAL_PRECISION:.0%}",
         n_p, m, s, bool(m >= UMBRAL_PRECISION), "logs/resultados_fase3.json -> normal.precision")
    m, s = ms(normal["recall"])
    fila("Recall complementario (escenario normal)", "recall_score(y_true, y_pred)",
         "informativo", n_p, m, s, None, "logs/resultados_fase3.json -> normal.recall")
    m, s = ms(normal["f1"])
    fila("F1-Score complementario (escenario normal)", "f1_score(y_true, y_pred)",
         f">= {UMBRAL_F1:.0%} (propuesto)", n_p, m, s, bool(m >= UMBRAL_F1),
         "logs/resultados_fase3.json -> normal.f1")
    m, s = ms(normal["auc_roc"])
    fila("AUC-ROC complementaria (escenario normal)", "roc_auc_score(y_true, probas)",
         f">= {UMBRAL_AUC} (propuesto)", n_p, m, s, bool(m >= UMBRAL_AUC),
         "logs/resultados_fase3.json -> normal.auc_roc")

    m, s = ms(sensor["precision"])
    fila("Precisión con sensor sin dato (30 % NaN en 3 variables)",
         "precision_score sobre el conjunto de prueba degradado", f">= {UMBRAL_PRECISION:.0%}",
         n_p, m, s, bool(m >= UMBRAL_PRECISION), "logs/resultados_fase3.json -> sensor_sin_dato.precision")
    m, s = ms(sensor["auc_roc"])
    fila("AUC-ROC con sensor sin dato", "roc_auc_score sobre el conjunto de prueba degradado",
         f">= {UMBRAL_AUC_ADVERSO}", n_p, m, s, bool(m >= UMBRAL_AUC_ADVERSO),
         "logs/resultados_fase3.json -> sensor_sin_dato.auc_roc")

    m, s = ms(reest["precision"])
    fila("Precisión bajo reestructuración organizacional (ingreso −15 %, satisfacción −1)",
         "precision_score sobre el conjunto de prueba degradado", f">= {UMBRAL_PRECISION:.0%}",
         n_p, m, s, bool(m >= UMBRAL_PRECISION), "logs/resultados_fase3.json -> reestructuracion.precision")
    m, s = ms(reest["auc_roc"])
    fila("AUC-ROC bajo reestructuración organizacional", "roc_auc_score sobre el conjunto de prueba degradado",
         f">= {UMBRAL_AUC_ADVERSO}", n_p, m, s, bool(m >= UMBRAL_AUC_ADVERSO),
         "logs/resultados_fase3.json -> reestructuracion.auc_roc")

    n_e = len(SEMILLAS_ESCENARIOS)
    pe = [p for p, _ in lim_extremo]
    m, s = ms(pe)
    fila("Caso límite: empleado con antigüedad máxima",
         "decidir(): 0 <= prob <= 1 y categoría coherente con los umbrales",
         "válido en las 5 repeticiones", n_e, m, s, all(v for _, v in lim_extremo),
         f"{sum(v for _, v in lim_extremo)}/{n_e} repeticiones válidas")
    pn = [p for p, _ in lim_nuevo]
    m, s = ms(pn)
    fila("Caso límite: empleado recién contratado",
         "decidir(): 0 <= prob <= 1 y categoría coherente con los umbrales",
         "válido en las 5 repeticiones", n_e, m, s, all(v for _, v in lim_nuevo),
         f"{sum(v for _, v in lim_nuevo)}/{n_e} repeticiones válidas")

    fila("Falla esperada: sensor sin dato (30 % NaN)", "decidir() procesa los 30 registros sin excepción",
         "éxito en las 5 repeticiones", n_e, None, None, all(falla_sensor),
         f"{sum(falla_sensor)}/{n_e} repeticiones sin error")
    fila("Falla esperada: lote de entrada vacío", "decidir() devuelve 0 registros sin excepción",
         "éxito en las 5 repeticiones", n_e, None, None, all(falla_vacio),
         f"{sum(falla_vacio)}/{n_e} repeticiones sin error")

    m, s = ms(exc_despues)
    ma, _ = ms(exc_antes)
    fila("Condición excepcional: reestructuración (probabilidad promedio, 100 empleados)",
         "prob_promedio(después) > prob_promedio(antes)", "aumento en las 5 repeticiones",
         n_e, m, s, all(exc_aumento),
         f"antes={ma}, después={m}; {sum(exc_aumento)}/{n_e} repeticiones con aumento")

    t_manual = 100 * MINUTOS_MANUAL_POR_EMPLEADO * 60
    red = [1 - (t / t_manual) for t in tiempos]
    m, s = ms(red)
    fila("Reducción de tiempo de procesamiento manual (100 empleados)",
         "1 - (tiempo_agente / tiempo_manual_estimado)",
         "informativo (línea base estimada, no medida en campo)", n_e, m, s, None,
         f"tiempo medio del agente = {np.mean(tiempos):.4f} s vs. {t_manual} s estimados")

    df = pd.DataFrame(filas)
    detalle = {
        "versiones": {
            "python": sys.version.split()[0],
            "scikit_learn": __import__("sklearn").__version__,
            "pandas": pd.__version__, "numpy": np.__version__,
        },
        "semillas_principal": SEMILLAS_PRINCIPAL,
        "semillas_escenarios": SEMILLAS_ESCENARIOS,
        "normal": {k: [round(v, 4) for v in vs] for k, vs in normal.items()},
        "sensor_sin_dato": {k: [round(v, 4) for v in vs] for k, vs in sensor.items()},
        "reestructuracion": {k: [round(v, 4) for v in vs] for k, vs in reest.items()},
        "tabla": filas,
    }
    for carpeta in (DIR_TESTS, DIR_LOGS):
        os.makedirs(carpeta, exist_ok=True)
        df.to_csv(os.path.join(carpeta, "resultados_fase3.csv"), index=False)
        with open(os.path.join(carpeta, "resultados_fase3.json"), "w", encoding="utf-8") as f:
            json.dump(detalle, f, ensure_ascii=False, indent=2, default=str)
    return df, detalle


if __name__ == "__main__":
    df, detalle = ejecutar_fase3()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 70)
    print(df[["medida", "umbral", "casos_ejecutados", "media", "desviacion_estandar", "cumple"]].to_string(index=False))
    print("\nVersiones:", detalle["versiones"])
    print("Resultados guardados en tests/ y en logs/ (resultados_fase3.csv y .json)")
