import json
from pathlib import Path

import numpy as np
from qcnn_ecg.entrenar import cargar_modelo
from qcnn_ecg.evaluacion import predecir_probabilidades
from qcnn_ecg.modelos import contar_parametros

from app.modelos.base import Clasificador


class ClasificadorCNN(Clasificador):
    clave = "cnn"
    id = "cnn1d"
    nombre = "CNN 1-D"

    def __init__(self, ruta: Path):
        self._modelo, self.umbral = cargar_modelo(ruta)
        self.parametros = contar_parametros(self._modelo)
        ficha = Path(ruta).with_suffix(".json")  # la escribe qcnn_ecg.desplegar
        if ficha.exists():
            datos = json.loads(ficha.read_text(encoding="utf-8"))
            self.id = datos.get("id", self.id)
            self.entrenado_con = datos.get("entrenado_con")
            self.metricas_prueba = datos.get("metricas_prueba")
            self.circuito = datos.get("circuito")

    def probabilidades(self, ventanas: np.ndarray, rr: np.ndarray | None = None) -> np.ndarray:
        return predecir_probabilidades(self._modelo, ventanas, rr=rr)  # ignora rr si no lo usa
