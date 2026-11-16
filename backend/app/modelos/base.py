from abc import ABC, abstractmethod

import numpy as np


class Clasificador(ABC):
    """Interfaz que debe cumplir cualquier modelo: simulado, CNN real o QCNN."""

    clave: str  # "cnn" o "qcnn": la clave que va en `resultados` del contrato
    id: str
    nombre: str
    parametros: int
    umbral: float
    entrenado_con: str | None = None
    metricas_prueba: dict | None = None

    @abstractmethod
    def probabilidades(self, ventanas: np.ndarray) -> np.ndarray:
        """Recibe (n, 288) y devuelve (n,) con la probabilidad de ANORMAL (0 a 1)."""