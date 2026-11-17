"""Pipeline único señal -> latidos: lo usan el entrenamiento, el backend y la QCNN.

remuestrear a 360 Hz -> detectar picos R (o usar los anotados) -> filtrar y normalizar toda la
señal -> recortar ventanas de 0.8 s -> rasgos de ritmo (RR). Si cambia un paso, cambia para todos.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from qcnn_ecg.config import FS, MEDIA_VENTANA, N_RASGOS_RR, RR_VENTANA_LOCAL
from qcnn_ecg.preprocesamiento import extraer_ventanas, preprocesar_senal

TOLERANCIA_FS_HZ = 0.5  # la fs deducida casi nunca es exactamente 360 (p. ej. 360.00001)
RR_MAXIMO = 5.0  # tope de los rasgos (pausas largas) para no desestabilizar la red


@dataclass
class Latidos:
    ventanas: np.ndarray  # (n, 288) de los latidos con ventana completa
    rr: np.ndarray  # (n, 4) float32, alineado con `ventanas`
    picos: np.ndarray  # TODOS los picos R usados, en muestras a 360 Hz
    mascara: np.ndarray  # (len(picos),) bool: qué picos tienen ventana completa
    remuestreada: bool


def remuestrear(senal: np.ndarray, fs_hz: float) -> tuple[np.ndarray, bool]:
    """Lleva la señal a 360 Hz. Devuelve (señal, si_se_remuestreó)."""
    if abs(fs_hz - FS) < TOLERANCIA_FS_HZ:
        return senal, False
    razon = Fraction(FS / fs_hz).limit_denominator(1000)
    return resample_poly(senal, razon.numerator, razon.denominator), True


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


def rasgos_rr(picos: np.ndarray, fs: int = FS) -> np.ndarray:
    """(n, 4): RR previo, RR posterior, RR local (media de los previos) y previo/posterior.

    Los tres primeros se dividen por el RR medio de la señal, así no dependen del ritmo base de
    cada persona. Donde no hay latido vecino (bordes) vale 1.0.
    # ponytail: el RR medio sale de toda la señal; en el entrenamiento son 30 min y en el backend
    # 10 s a 5 min. Si pesa, usar una mediana móvil.
    """
    n = len(picos)
    rasgos = np.ones((n, N_RASGOS_RR))
    if n < 2:
        return rasgos.astype(np.float32)
    rr = np.diff(picos) / fs
    rel = rr / rr.mean()
    rasgos[1:, 0] = rel  # previo
    rasgos[:-1, 1] = rel  # posterior
    acumulado = np.concatenate([[0.0], np.cumsum(rel)])
    for i in range(1, n):  # local: media de hasta RR_VENTANA_LOCAL RR previos
        ini = max(0, i - RR_VENTANA_LOCAL)
        rasgos[i, 2] = (acumulado[i] - acumulado[ini]) / (i - ini)
    rasgos[1:-1, 3] = rasgos[1:-1, 0] / rasgos[1:-1, 1]
    return np.clip(rasgos, 0.0, RR_MAXIMO).astype(np.float32)


def procesar_senal(
    senal: np.ndarray, fs_hz: float, picos: np.ndarray | None = None, detector=detectar_picos_r
) -> Latidos:
    """Señal cruda (mV) -> ventanas y rasgos RR. `picos` (a 360 Hz) evita la detección."""
    x, remuestreada = remuestrear(senal, fs_hz)
    picos = np.asarray(detector(x) if picos is None else picos, dtype=np.int64)
    ventanas, mascara = extraer_ventanas(preprocesar_senal(x, fs=FS), picos, MEDIA_VENTANA)
    return Latidos(ventanas, rasgos_rr(picos)[mascara], picos, mascara, remuestreada)
