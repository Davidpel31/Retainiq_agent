# -*- coding: utf-8 -*-
"""
Componente de Memoria del RETAINIQ-AGENT (Sección 8.3 del diseño).

Implementa memoria de corto plazo (datos y predicciones del ciclo mensual
actual) y memoria de largo plazo (histórico de predicciones y resultados
reales, usado para auditoría y para el componente de aprendizaje).
"""
import json
import os
from datetime import datetime


class MemoriaAgente:
    """Estado interno del agente: lo que lo distingue de un agente de
    reflejo simple, según la Sección 2 del diseño (agente basado en
    utilidad con capacidades de aprendizaje)."""

    def __init__(self, ruta_largo_plazo="logs/memoria_largo_plazo.json"):
        self.ruta_largo_plazo = ruta_largo_plazo
        self.corto_plazo = {}  # predicciones del ciclo mensual actual
        self.largo_plazo = self._cargar_largo_plazo()

    def _cargar_largo_plazo(self):
        if os.path.exists(self.ruta_largo_plazo):
            with open(self.ruta_largo_plazo, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"ciclos": [], "resultados_reales": []}

    def guardar_ciclo(self, ciclo_id, predicciones_df, metricas):
        """Registra un ciclo mensual completo en memoria de corto y
        largo plazo (Sección 8.2, paso 7: 'registro en memoria')."""
        self.corto_plazo = {
            "ciclo_id": ciclo_id,
            "timestamp": datetime.utcnow().isoformat(),
            "n_empleados": len(predicciones_df),
            "metricas": metricas,
        }
        self.largo_plazo["ciclos"].append(self.corto_plazo)
        os.makedirs(os.path.dirname(self.ruta_largo_plazo), exist_ok=True)
        with open(self.ruta_largo_plazo, "w", encoding="utf-8") as f:
            json.dump(self.largo_plazo, f, ensure_ascii=False, indent=2)

    def registrar_resultado_real(self, empleado_id, prediccion, resultado_real):
        """Registra si la predicción coincidió con lo ocurrido realmente
        (renunció / permaneció), insumo del componente de aprendizaje."""
        self.largo_plazo["resultados_reales"].append({
            "empleado_id": empleado_id,
            "prediccion": prediccion,
            "resultado_real": resultado_real,
            "timestamp": datetime.utcnow().isoformat(),
        })
        with open(self.ruta_largo_plazo, "w", encoding="utf-8") as f:
            json.dump(self.largo_plazo, f, ensure_ascii=False, indent=2)

    def historico_ciclos(self):
        return self.largo_plazo["ciclos"]
