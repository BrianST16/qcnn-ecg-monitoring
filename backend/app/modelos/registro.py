import logging

from app.config import Ajustes
from app.modelos.base import Clasificador
from app.modelos.simulado import CNNSimulada, QCNNMaqueta

log = logging.getLogger("uvicorn.error")


def _cargar_cnn(ajustes: Ajustes) -> Clasificador:
    if ajustes.modelo_simulado:
        log.warning("MODELO SIMULADO activo: las clasificaciones de la CNN NO son reales.")
        return CNNSimulada()
    # Imports aquí dentro: en modo simulado el servidor arranca rápido y sin cargar PyTorch.
    import torch

    from app.modelos.cnn import ClasificadorCNN

    torch.set_num_threads(ajustes.torch_hilos)
    return ClasificadorCNN(ajustes.ruta_modelo_cnn)


def _cargar_qcnn(ajustes: Ajustes) -> Clasificador | None:
    if ajustes.qcnn_modo == "desactivada":
        return None  # el análisis responde "no_disponible" para la QCNN
    if ajustes.qcnn_modo == "maqueta":
        log.warning("QCNN en modo MAQUETA: las clasificaciones de la QCNN NO son reales.")
        return QCNNMaqueta()
    from app.modelos.qcnn import ClasificadorQCNN

    return ClasificadorQCNN(ajustes.ruta_modelo_qcnn)


def cargar_modelos(ajustes: Ajustes) -> dict[str, Clasificador]:
    modelos: dict[str, Clasificador] = {"cnn": _cargar_cnn(ajustes)}
    qcnn = _cargar_qcnn(ajustes)
    if qcnn is not None:
        modelos["qcnn"] = qcnn
    return modelos