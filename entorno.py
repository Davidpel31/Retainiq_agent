# -*- coding: utf-8 -*-
"""
Simulador del entorno del RETAINIQ-AGENT (Sección 3 y 4 del diseño).

Genera las percepciones descritas en la Sección 5 (Entradas y sensores)
a partir del dataset IBM HR Analytics, y respeta las propiedades de
ambiente declaradas en la Sección 3:
  - Parcialmente observable: solo se exponen las variables del dataset,
    no factores externos (ofertas de la competencia, situación personal).
  - Estocástico: se inyecta ruido aleatorio en variables sensibles al
    tiempo (satisfacción, horas extra) para simular que el entorno
    cambia entre ciclos mensuales.
  - Dinámico: cada ciclo mensual puede alterar condiciones
    organizacionales (p. ej. una ronda de ajuste salarial simulada).
  - Secuencial: el estado de cada empleado en el ciclo t influye en el
    ciclo t+1 (si fue intervenido, su satisfacción puede mejorar).
"""
import numpy as np
import pandas as pd


class EntornoRRHH:
    def __init__(self, ruta_csv="data/HR-Employee-Attrition.csv", seed=42):
        self.df_base = pd.read_csv(ruta_csv)
        self.rng = np.random.default_rng(seed)
        # Estado mutable que evoluciona entre ciclos (dinamismo/secuencialidad)
        self.estado_actual = self.df_base.copy()
        self.ciclo = 0

    def _inyectar_estocasticidad(self, df):
        """El agente opera en un entorno estocástico: la satisfacción
        laboral y las horas extra fluctúan de un mes a otro sin que el
        agente pueda predecirlas con certeza."""
        df = df.copy()
        ruido_satisfaccion = self.rng.integers(-1, 2, size=len(df))
        df["JobSatisfaction"] = (df["JobSatisfaction"] + ruido_satisfaccion).clip(1, 4)
        # 5% de probabilidad de que un empleado entre en horas extra este mes
        cambio_overtime = self.rng.random(len(df)) < 0.05
        df.loc[cambio_overtime, "OverTime"] = df.loc[cambio_overtime, "OverTime"].map(
            {"Yes": "No", "No": "Yes"}
        )
        return df

    def _aplicar_intervencion(self, df, ids_intervenidos):
        """Entorno secuencial: una intervención de RR.HH. en el ciclo
        actual puede mejorar el perfil de riesgo en el ciclo siguiente."""
        df = df.copy()
        mascara = df["EmployeeNumber"].isin(ids_intervenidos)
        mejora = self.rng.random(mascara.sum()) < 0.4  # 40% de intervenciones exitosas
        idx_mejora = df.loc[mascara].index[mejora]
        df.loc[idx_mejora, "JobSatisfaction"] = (
            df.loc[idx_mejora, "JobSatisfaction"] + 1
        ).clip(upper=4)
        return df

    def observar(self, ids_intervenidos_mes_anterior=None):
        """Sensor principal: entrega el dataset estructurado del ciclo
        actual (Sección 5 y Sección 8.1 — Percepción y manejo de entradas).
        Devuelve una copia; el agente NO tiene acceso al CSV directamente,
        solo a lo que este método expone (observabilidad parcial)."""
        self.ciclo += 1
        df = self.estado_actual.copy()
        if ids_intervenidos_mes_anterior:
            df = self._aplicar_intervencion(df, ids_intervenidos_mes_anterior)
        df = self._inyectar_estocasticidad(df)
        self.estado_actual = df
        return df.drop(columns=["Attrition"]), df["Attrition"]  # percepción, ground truth oculto

    def inyectar_fallo_sensor(self, df, columnas, fraccion=0.1):
        """Simula un caso de falla esperada: sensor sin dato (requerido
        por la Fase 3 — casos de falla esperada)."""
        df = df.copy()
        n = int(len(df) * fraccion)
        idx = self.rng.choice(df.index, size=n, replace=False)
        for col in columnas:
            df.loc[idx, col] = np.nan
        return df

    def reestructuracion_organizacional(self, df):
        """Condición excepcional declarada en la Sección 4: alteración
        drástica del entorno (ronda de recortes salariales)."""
        df = df.copy()
        df["MonthlyIncome"] = (df["MonthlyIncome"] * 0.85).round(0)
        df["JobSatisfaction"] = (df["JobSatisfaction"] - 1).clip(1, 4)
        return df
