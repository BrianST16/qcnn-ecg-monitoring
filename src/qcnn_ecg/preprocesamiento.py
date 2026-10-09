"""Filtrado, normalización, etiquetado y segmentación latido a latido.

Reproduce el preprocesamiento del notebook EDA original (Butterworth de orden 4
0.5-40 Hz con filtfilt, z-score por registro y ventana de ±0.4 s alrededor de R).
"""

import numpy as np
from scipy.signal import butter, filtfilt

from qcnn_ecg.config import (
    BANDA_HZ,
    FS,
    MEDIA_VENTANA,
    ORDEN_FILTRO,
    SIMBOLOS_ANORMAL,
    SIMBOLOS_NORMAL,
)


def filtrar_pasabanda(
    senal: np.ndarray,
    fs: int = FS,
    banda: tuple[float, float] = BANDA_HZ,
    orden: int = ORDEN_FILTRO,
) -> np.ndarray:
    """Filtro Butterworth pasa-banda de fase cero (no causal, para análisis offline)."""
    b, a = butter(orden, banda, btype="band", fs=fs)
    return filtfilt(b, a, senal)


def normalizar_registro(senal: np.ndarray) -> np.ndarray:
    """Z-score con la media y desviación del propio registro."""
    desviacion = senal.std()
    if desviacion == 0:
        return senal - senal.mean()
    return (senal - senal.mean()) / desviacion


def preprocesar_senal(senal: np.ndarray, fs: int = FS) -> np.ndarray:
    """Filtrado pasa-banda seguido de normalización por registro."""
    return normalizar_registro(filtrar_pasabanda(senal, fs=fs))


def etiquetar_simbolo(simbolo: str) -> int | None:
    """0 = normal, 1 = anormal, None = se descarta (Q, marcapasos, no-latido)."""
    if simbolo in SIMBOLOS_NORMAL:
        return 0
    if simbolo in SIMBOLOS_ANORMAL:
        return 1
    return None


def extraer_ventanas(
    senal: np.ndarray, muestras: np.ndarray, media_ventana: int = MEDIA_VENTANA
) -> tuple[np.ndarray, np.ndarray]:
    """Recorta una ventana de 2*media_ventana muestras centrada en cada posición.

    Devuelve (ventanas, mascara): `mascara` indica qué posiciones tenían la
    ventana completa dentro de la señal; las de los bordes se descartan.
    """
    muestras = np.asarray(muestras, dtype=np.int64)
    mascara = (muestras - media_ventana >= 0) & (muestras + media_ventana <= len(senal))
    centros = muestras[mascara]
    indices = centros[:, None] + np.arange(-media_ventana, media_ventana)[None, :]
    return senal[indices], mascara
