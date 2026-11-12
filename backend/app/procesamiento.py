from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from qcnn_ecg.config import FS, MEDIA_VENTANA
from qcnn_ecg.preprocesamiento import extraer_ventanas, preprocesar_senal
from scipy.signal import resample_poly

from app.csv_ecg import SenalLeida
from app.errores import ErrorAPI

MIN_LATIDOS = 5
TOLERANCIA_FS_HZ = 0.5  # la fs deducida casi nunca es exactamente 360 (p. ej. 360.00001)


@dataclass
class SenalProcesada:
    ventanas: np.ndarray  # (latidos_evaluados, 288), lista para el modelo
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


def remuestrear_a_modelo(mv: np.ndarray, fs_hz: float) -> tuple[np.ndarray, bool]:
    """Lleva la señal a 360 Hz. Devuelve (señal, si_se_remuestreó)."""
    if abs(fs_hz - FS) < TOLERANCIA_FS_HZ:
        return mv, False
    razon = Fraction(FS / fs_hz).limit_denominator(1000)
    return resample_poly(mv, razon.numerator, razon.denominator), True


def detectar_picos_r(senal_360: np.ndarray) -> np.ndarray:
    """Picos R con XQRS de wfdb, corregidos sobre la señal cruda (contrato, sección 4)."""
    from wfdb import processing  # import perezoso: es pesado y solo hace falta aquí

    picos = processing.xqrs_detect(sig=senal_360, fs=FS, verbose=False)
    if len(picos) == 0:
        return np.array([], dtype=np.int64)
    picos = processing.correct_peaks(
        senal_360, peak_inds=picos, search_radius=18, smooth_window_size=7
    )
    return np.unique(np.asarray(picos, dtype=np.int64))


def procesar_senal(
    senal: SenalLeida,
    detector: Callable[[np.ndarray], np.ndarray] = detectar_picos_r,
) -> SenalProcesada:
    """Pasos 2 a 5 del contrato: remuestrear, detectar R, filtrar/normalizar y recortar ventanas."""
    x, remuestreada = remuestrear_a_modelo(senal.mv, senal.fs_hz)

    picos = np.asarray(detector(x), dtype=np.int64)
    if len(picos) < MIN_LATIDOS:
        raise ErrorAPI(
            "SIN_LATIDOS_SUFICIENTES",
            f"Se detectaron {len(picos)} latidos y se necesitan al menos {MIN_LATIDOS}.",
            {"latidos_detectados": int(len(picos)), "minimo": MIN_LATIDOS},
        )

    x_filtrada = preprocesar_senal(x, fs=FS)  # filtro y z-score sobre TODA la señal
    ventanas, mascara = extraer_ventanas(x_filtrada, picos, MEDIA_VENTANA)
    if len(ventanas) == 0:
        raise ErrorAPI(
            "SIN_LATIDOS_SUFICIENTES",
            "Ningún latido tiene su ventana completa dentro de la señal.",
            {"latidos_detectados": int(len(picos)), "minimo": MIN_LATIDOS},
        )

    rr = np.diff(picos) / FS  # todos los RR, incluidos los latidos descartados por el borde
    rr_previo = np.concatenate([[np.nan], rr])[mascara]
    tiempos = float(senal.tiempo_s[0]) + picos[mascara] / FS

    return SenalProcesada(
        ventanas=ventanas,
        tiempos_s=tiempos,
        rr_previo_s=rr_previo,
        latidos_detectados=int(len(picos)),
        fc_media_lpm=float(60.0 / rr.mean()),
        fs_original_hz=senal.fs_hz,
        remuestreada=remuestreada,
    )