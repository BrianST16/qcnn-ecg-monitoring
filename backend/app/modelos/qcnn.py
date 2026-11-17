from pathlib import Path

import numpy as np

from app.modelos.base import Clasificador


class ClasificadorQCNN(Clasificador):
    """Adaptador de la QCNN real: lo ÚNICO que hay que rellenar cuando el modelo esté listo.

    Contrato con quien entrena la QCNN:
      - Entrada: `ventanas`, un array (n, 288) de float, ya filtrado (0.5-40 Hz) y con z-score,
        igual que recibe la CNN.
      - Salida: un array (n,) con la probabilidad de ANORMAL de cada latido, entre 0 y 1,
        en el mismo orden de las ventanas.
      - Atributos: id (texto), nombre, parametros (entero), umbral (0 a 1).
    """

    clave = "qcnn"
    id = "qcnn-base-v1"  # TODO: poner el identificador real y versionado del modelo
    nombre = "QCNN (simulada)"  # simulada = circuito cuántico ejecutado en un simulador

    def __init__(self, ruta: Path):
        # TODO 1 de 2: cargar el modelo entrenado desde `ruta` (una sola vez, al iniciar).
        # Ejemplo de lo que debe quedar al terminar este método:
        #   self._modelo = ...           # el circuito o red ya cargados
        #   self.umbral = ...            # el guardado junto al modelo
        #   self.parametros = ...        # número de parámetros entrenables
        raise NotImplementedError(
            "La QCNN real aún no está conectada: completa ClasificadorQCNN en app/modelos/qcnn.py."
        )

    def probabilidades(self, ventanas: np.ndarray, rr: np.ndarray | None = None) -> np.ndarray:
        # TODO 2 de 2: devolver un array (n,) de probabilidades de ANORMAL.
        raise NotImplementedError