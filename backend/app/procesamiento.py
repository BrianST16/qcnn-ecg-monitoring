from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from qcnn_ecg import pipeline
from qcnn_ecg.config import FS
from qcnn_ecg.pipeline import detectar_picos_r
from qcnn_ecg.pipeline import remuestrear as remuestrear_a_modelo  # noqa: F401 (lo usan las pruebas)

from app.csv_ecg import SenalLeida
from app.errores import ErrorAPI

MIN_LATIDOS = 5


@dataclass
class SenalProcesada:
    ventanas: np.ndarray  # (latidos_evaluados, 288), lista para el modelo
    rr: np.ndarray  # (latidos_evaluados, 4) rasgos de ritmo para los modelos que los usan
    tiempos_s: np.ndarray  # posición del pico R de cada latido evaluado, en la escala del CSV
    rr_previo_s: np.ndarray  # RR respecto al latido detectado anterior; NaN si no hay anterior
    latidos_detectados: int
    fc_media_lpm: float
    fs_original_hz: float
    remuestreada: bool

    @property
    def latidos_evaluados(self) -> int:
        return len(self.ventanas)

    @property
    def latidos_descartados_borde(self) -> int:
        return self.latidos_detectados - self.latidos_evaluados


def procesar_senal(
    senal: SenalLeida,
    detector: Callable[[np.ndarray], np.ndarray] = detectar_picos_r,
) -> SenalProcesada:
    """Pasos 2 a 5 del contrato, delegados en qcnn_ecg.pipeline (el mismo código del entrenamiento)."""
    lat = pipeline.procesar_senal(senal.mv, senal.fs_hz, detector=detector)
    picos = lat.picos
    if len(picos) < MIN_LATIDOS:
        raise ErrorAPI(
            "SIN_LATIDOS_SUFICIENTES",
            f"Se detectaron {len(picos)} latidos y se necesitan al menos {MIN_LATIDOS}.",
            {"latidos_detectados": int(len(picos)), "minimo": MIN_LATIDOS},
        )
    if len(lat.ventanas) == 0:
        raise ErrorAPI(
            "SIN_LATIDOS_SUFICIENTES",
            "Ningún latido tiene su ventana completa dentro de la señal.",
            {"latidos_detectados": int(len(picos)), "minimo": MIN_LATIDOS},
        )

    rr = np.diff(picos) / FS  # todos los RR, incluidos los latidos descartados por el borde
    rr_previo = np.concatenate([[np.nan], rr])[lat.mascara]
    tiempos = float(senal.tiempo_s[0]) + picos[lat.mascara] / FS

    return SenalProcesada(
        ventanas=lat.ventanas,
        rr=lat.rr,
        tiempos_s=tiempos,
        rr_previo_s=rr_previo,
        latidos_detectados=int(len(picos)),
        fc_media_lpm=float(60.0 / rr.mean()),
        fs_original_hz=senal.fs_hz,
        remuestreada=lat.remuestreada,
    )