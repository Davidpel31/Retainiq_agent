# -*- coding: utf-8 -*-
"""
RETAINIQ-AGENT — Implementación del agente de predicción de rotación
de empleados, siguiendo el diseño conceptual (AIMA: agente basado en
utilidad con capacidades de aprendizaje) y los 7 componentes de la
arquitectura de IA agéntica de IBM (Sección 8 del documento de diseño).

Cada método de esta clase está explícitamente etiquetado con el
componente IBM que implementa, y el ciclo percepción -> decisión ->
acción puede identificarse siguiendo los métodos `percibir`,
`decidir` y `actuar` dentro de `ejecutar_ciclo_mensual`.
"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

from memoria import MemoriaAgente

COLUMNAS_CATEGORICAS = [
    "BusinessTravel", "Department", "EducationField", "Gender",
    "JobRole", "MaritalStatus", "Over18", "OverTime",
]
COLUMNAS_NUMERICAS = [
    "Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]


class RetainIQAgent:
    """Agente basado en utilidad con capacidades de aprendizaje
    (Sección 2 del diseño). Mantiene estado interno (memoria) y una
    función de utilidad que prioriza verdaderos positivos (detectar
    empleados en riesgo real) sobre falsos positivos, según la
    Sección 8.4 del diseño."""

    UMBRAL_ALTO = 0.70
    UMBRAL_MEDIO = 0.40

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.memoria = MemoriaAgente(os.path.join(log_dir, "memoria_largo_plazo.json"))
        self.modelo = None
        self.encoders = {}
        self._log_ejecucion = []

    # ------------------------------------------------------------------
    # Componente 8.1 — Percepción y manejo de entradas
    # ------------------------------------------------------------------
    def percibir(self, df_crudo):
        """Valida integridad, imputa datos faltantes, normaliza y
        codifica variables categóricas. Devuelve un dataset listo
        para el modelo (Sección 8.1)."""
        df = df_crudo.copy()
        n_faltantes = df.isna().sum().sum()

        # Imputación simple: numéricas -> mediana, categóricas -> moda
        for col in COLUMNAS_NUMERICAS:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        for col in COLUMNAS_CATEGORICAS:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0])

        # Variable derivada enriquecida (ejemplo mencionado en Sección 8.1)
        if "YearsAtCompany" in df.columns and "NumCompaniesWorked" in df.columns:
            df["RotacionPrevia"] = df["NumCompaniesWorked"] / (df["YearsAtCompany"] + 1)

        # Codificación de categóricas
        df_cod = df.copy()
        for col in COLUMNAS_CATEGORICAS:
            if col not in df_cod.columns:
                continue
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df_cod[col] = self.encoders[col].fit_transform(df_cod[col].astype(str))
            else:
                le = self.encoders[col]
                df_cod[col] = df_cod[col].astype(str).map(
                    lambda v: v if v in le.classes_ else le.classes_[0]
                )
                df_cod[col] = le.transform(df_cod[col])

        self._registrar_evento("percepcion", {
            "n_registros": len(df), "n_valores_imputados": int(n_faltantes),
        })
        return df_cod

    # ------------------------------------------------------------------
    # Componente 8.4 — Razonamiento y toma de decisiones
    # ------------------------------------------------------------------
    def entrenar(self, df_crudo, y, test_size=0.2, seed=42):
        """Entrena el modelo de Random Forest sobre el histórico
        etiquetado. Corresponde al estado inicial del componente de
        razonamiento (Sección 8.4)."""
        X = self.percibir(df_crudo)
        feats = [c for c in COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS + ["RotacionPrevia"]
                  if c in X.columns]
        self.features = feats
        y_bin = (y == "Yes").astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X[feats], y_bin, test_size=test_size, random_state=seed, stratify=y_bin
        )
        self.modelo = RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=seed
        )
        self.modelo.fit(X_train, y_train)

        y_prob = self.modelo.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metricas = {
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "auc_roc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "n_train": len(X_train), "n_test": len(X_test),
        }
        self._registrar_evento("entrenamiento_inicial", metricas)
        return metricas

    def decidir(self, df_crudo):
        """Produce, para cada empleado, una probabilidad de renuncia y
        una categoría de riesgo (alto/medio/bajo), aplicando la función
        de utilidad descrita en la Sección 8.4: prioriza no dejar pasar
        verdaderos positivos (recall) sobre evitar falsos positivos."""
        if self.modelo is None:
            raise RuntimeError("El agente debe entrenarse (entrenar()) antes de decidir.")
        if len(df_crudo) == 0:
            # Manejo explícito de lote vacío: el agente no falla, informa
            # que no hay nada que decidir en este ciclo.
            self._registrar_evento("decision", {"n_evaluados": 0, "n_alto_riesgo": 0,
                                                  "advertencia": "lote de entrada vacío"})
            return pd.DataFrame(columns=["EmployeeNumber", "probabilidad_renuncia", "categoria_riesgo"])
        X = self.percibir(df_crudo)
        X = X.reindex(columns=self.features, fill_value=0)
        probas = self.modelo.predict_proba(X)[:, 1]

        categorias = np.where(
            probas >= self.UMBRAL_ALTO, "alto",
            np.where(probas >= self.UMBRAL_MEDIO, "medio", "bajo"),
        )
        resultado = pd.DataFrame({
            "EmployeeNumber": df_crudo["EmployeeNumber"].values,
            "probabilidad_renuncia": probas.round(4),
            "categoria_riesgo": categorias,
        })
        self._registrar_evento("decision", {
            "n_evaluados": len(resultado),
            "n_alto_riesgo": int((categorias == "alto").sum()),
        })
        return resultado

    # ------------------------------------------------------------------
    # Componente 8.5 — Llamada de herramientas y acción (tool calling)
    # ------------------------------------------------------------------
    def actuar(self, decisiones_df, ciclo_id):
        """Ejecuta las acciones externas descritas en la Sección 6:
        escribir el reporte (simulado como archivo CSV/JSON — 'dashboard'),
        notificar a líderes por correo (simulado como entrada de log) y
        actualizar el módulo de ML. Estas llamadas representan las
        integraciones con HRIS/nómina/dashboard de la Sección 8.5."""
        ruta_reporte = os.path.join(self.log_dir, f"reporte_ciclo_{ciclo_id}.csv")
        decisiones_df.to_csv(ruta_reporte, index=False)

        alto_riesgo = decisiones_df[decisiones_df["categoria_riesgo"] == "alto"]
        notificacion = {
            "accion": "notificar_lideres_correo_electronico",
            "destinatarios_simulados": len(alto_riesgo),
            "empleados_notificados": alto_riesgo["EmployeeNumber"].tolist(),
        }
        ruta_notificacion = os.path.join(self.log_dir, f"notificacion_ciclo_{ciclo_id}.json")
        with open(ruta_notificacion, "w", encoding="utf-8") as f:
            json.dump(notificacion, f, ensure_ascii=False, indent=2)

        self._registrar_evento("accion", {
            "reporte": ruta_reporte, "notificacion": ruta_notificacion,
            "n_alto_riesgo_notificados": len(alto_riesgo),
        })
        return {"reporte": ruta_reporte, "notificacion": ruta_notificacion,
                "ids_alto_riesgo": alto_riesgo["EmployeeNumber"].tolist()}

    # ------------------------------------------------------------------
    # Componente 8.6 — Comunicación
    # ------------------------------------------------------------------
    def comunicar_resumen(self, decisiones_df, ciclo_id):
        """Genera el resumen estructurado que acompaña el dashboard
        (Sección 8.6)."""
        resumen = {
            "ciclo_id": ciclo_id,
            "total_empleados": len(decisiones_df),
            "riesgo_alto": int((decisiones_df["categoria_riesgo"] == "alto").sum()),
            "riesgo_medio": int((decisiones_df["categoria_riesgo"] == "medio").sum()),
            "riesgo_bajo": int((decisiones_df["categoria_riesgo"] == "bajo").sum()),
            "probabilidad_promedio": round(float(decisiones_df["probabilidad_renuncia"].mean()), 4),
        }
        self._registrar_evento("comunicacion", resumen)
        return resumen

    # ------------------------------------------------------------------
    # Componente 8.7 — Aprendizaje y adaptación
    # ------------------------------------------------------------------
    def recalibrar(self, df_crudo_actualizado, y_actualizado, seed=42):
        """Reentrena el modelo incorporando los resultados reales
        observados, simulando la recalibración trimestral de la
        Sección 8.7."""
        metricas = self.entrenar(df_crudo_actualizado, y_actualizado, seed=seed)
        self._registrar_evento("recalibracion", metricas)
        return metricas

    # ------------------------------------------------------------------
    # Ciclo completo — Componente 8.2 (Planificación y descomposición)
    # ------------------------------------------------------------------
    def ejecutar_ciclo_mensual(self, df_crudo, ciclo_id):
        """Orquesta el flujo mensual de 7 pasos descrito en la
        Sección 8.2: (1) programación [implícita], (2) percepción,
        (3) preprocesamiento [dentro de percibir], (4) decisión,
        (5) generación de reporte [comunicar], (6) distribución [actuar],
        (7) registro en memoria."""
        decisiones = self.decidir(df_crudo)                 # 2-4: percepción+decisión
        resumen = self.comunicar_resumen(decisiones, ciclo_id)  # 5: comunicación
        resultado_accion = self.actuar(decisiones, ciclo_id)    # 6: acción
        self.memoria.guardar_ciclo(ciclo_id, decisiones, resumen)  # 7: memoria
        return {
            "decisiones": decisiones, "resumen": resumen,
            "accion": resultado_accion,
        }

    # ------------------------------------------------------------------
    # Utilidades de logging (Fase 2: registro de ejecución con timestamp)
    # ------------------------------------------------------------------
    def _registrar_evento(self, tipo, detalle):
        evento = {
            "timestamp": datetime.utcnow().isoformat(),
            "tipo": tipo,
            "detalle": detalle,
        }
        self._log_ejecucion.append(evento)

    def volcar_log(self, ruta=None):
        ruta = ruta or os.path.join(self.log_dir, "log_ejecucion.json")
        existentes = []
        if os.path.exists(ruta):
            with open(ruta, "r", encoding="utf-8") as f:
                existentes = json.load(f)
        existentes.extend(self._log_ejecucion)
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(existentes, f, ensure_ascii=False, indent=2)
        self._log_ejecucion = []
        return ruta
