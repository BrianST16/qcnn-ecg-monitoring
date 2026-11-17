from pathlib import Path

import numpy as np
from qcnn_ecg.entrenar import cargar_modelo
from qcnn_ecg.evaluacion import predecir_probabilidades
from qcnn_ecg.modelos import contar_parametros

from app.modelos.base import Clasificador


class ClasificadorCNN(Clasificador):
    clave = "cnn"
    id = "cnn1d-base-v1"
    nombre = "CNN 1-D"

    def __init__(self, ruta: Path):
        self._modelo, self.umbral = cargar_modelo(ruta)
        self.parametros = contar_parametros(self._modelo)

    def probabilidades(self, ventanas: np.ndarray, rr: np.ndarray | None = None) -> np.ndarray:
        return predecir_probabilidades(self._modelo, ventanas, rr=rr)  # ignora rr si no lo usa