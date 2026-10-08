"""Configuración común de las pruebas."""

import pytest

from qcnn_ecg import config

requiere_mitdb = pytest.mark.skipif(
    not (config.DIR_MITDB / "100.dat").exists(),
    reason="MIT-BIH no está descargada (uv run python -m qcnn_ecg.preparar_datos)",
)
requiere_conjunto = pytest.mark.skipif(
    not (config.DIR_PROCESADOS / config.NOMBRE_CONJUNTO / "X.npy").exists(),
    reason="Conjunto procesado no construido (uv run python -m qcnn_ecg.preparar_datos)",
)
