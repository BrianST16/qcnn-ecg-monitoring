import numpy as np

from app.modelos.base import Clasificador


class CNNSimulada(Clasificador):
    clave = "cnn"
    id = "cnn1d-simulada"
    nombre = "CNN 1-D (SIMULADA)"
    parametros = 0
    umbral = 0.615

    def probabilidades(self, ventanas: np.ndarray, rr: np.ndarray | None = None) -> np.ndarray:
        n = len(ventanas)
        rng = np.random.default_rng(42)
        es_anormal = rng.random(n) < 0.15
        p_anormal = rng.uniform(0.70, 0.99, n)
        p_normal = rng.uniform(0.01, 0.50, n)
        return np.where(es_anormal, p_anormal, p_normal).astype(np.float32)
    
class QCNNMaqueta(Clasificador):
    """Maqueta de la QCNN: probabilidades inventadas, solo para probar el flujo de dos modelos."""

    clave = "qcnn"
    id = "qcnn-maqueta"
    nombre = "QCNN (MAQUETA)"
    parametros = 0
    umbral = 0.5

    def probabilidades(self, ventanas, rr=None):
        rng = np.random.default_rng(7)
        n = len(ventanas)
        anormal = rng.random(n) < 0.20
        return np.where(anormal, rng.uniform(0.55, 0.98, n), rng.uniform(0.02, 0.45, n))