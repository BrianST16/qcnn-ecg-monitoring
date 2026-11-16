from dataclasses import dataclass
from io import StringIO

import numpy as np
import pandas as pd

from app.errores import ErrorAPI

ENCABEZADO = "tiempo_s,mlii_mv"
FS_MIN_HZ, FS_MAX_HZ = 100.0, 1000.0
DURACION_MIN_S, DURACION_MAX_S = 10.0, 300.0
AMPLITUD_MAX_MV = 10.0
TOLERANCIA_PASO = 0.01  # 1 %


@dataclass
class SenalLeida:
    tiempo_s: np.ndarray
    mv: np.ndarray
    fs_hz: float
    duracion_s: float

    @property
    def muestras(self) -> int:
        return len(self.mv)


def _linea(indice: int) -> int:
    """Número de línea del archivo para la posición `indice` de los datos (la línea 1 es el encabezado)."""
    return indice + 2


def leer_senal(texto: str) -> SenalLeida:
    """Lee y valida el CSV según la sección 2 del contrato. Lanza ErrorAPI si algo falla."""
    # 1. Encabezado exacto
    primera_linea = texto.split("\n", 1)[0].strip()
    if primera_linea != ENCABEZADO:
        raise ErrorAPI(
            "COLUMNAS_INVALIDAS",
            f"El encabezado debe ser exactamente «{ENCABEZADO}».",
            {"encabezado_esperado": ENCABEZADO, "encabezado_recibido": primera_linea[:100]},
        )

    # 2. Valores numéricos y sin vacíos
    try:
        tabla = pd.read_csv(
            StringIO(texto),
            header=0,
            names=["tiempo_s", "mlii_mv"],
            dtype=str,
            keep_default_na=False,
            index_col=False,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        raise ErrorAPI(
            "VALORES_INVALIDOS", "Alguna fila no tiene exactamente dos columnas."
        ) from None

    if tabla.empty:
        raise ErrorAPI("VALORES_INVALIDOS", "El archivo no contiene filas de datos.")

    tiempo = pd.to_numeric(tabla["tiempo_s"], errors="coerce").to_numpy(dtype=float)
    mv = pd.to_numeric(tabla["mlii_mv"], errors="coerce").to_numpy(dtype=float)
    invalidos = ~(np.isfinite(tiempo) & np.isfinite(mv))
    if invalidos.any():
        linea = _linea(int(np.argmax(invalidos)))
        raise ErrorAPI(
            "VALORES_INVALIDOS",
            f"La fila {linea} tiene un valor vacío o no numérico.",
            {"fila": linea, "filas_invalidas": int(invalidos.sum())},
        )

    n = len(tiempo)
    if n < 2:
        raise ErrorAPI(
            "DURACION_INSUFICIENTE",
            "El archivo tiene menos de 2 filas de datos.",
            {"duracion_s": 0.0, "minimo_s": DURACION_MIN_S, "maximo_s": DURACION_MAX_S},
        )

    # 3. Tiempo estrictamente creciente
    pasos = np.diff(tiempo)
    if np.any(pasos <= 0):
        linea = _linea(int(np.argmax(pasos <= 0)) + 1)
        raise ErrorAPI(
            "TIEMPO_NO_CRECIENTE",
            f"El tiempo no crece en la fila {linea}.",
            {"fila": linea},
        )

    # 4. Paso de tiempo constante (desviación máxima < 1 %)
    paso_medio = float(pasos.mean())
    desviacion = float(np.max(np.abs(pasos - paso_medio)) / paso_medio)
    if desviacion >= TOLERANCIA_PASO:
        raise ErrorAPI(
            "MUESTREO_IRREGULAR",
            f"El paso de tiempo no es constante: varía hasta {desviacion * 100:.1f} % y el límite es 1 %.",
            {"desviacion_max_pct": round(desviacion * 100, 2), "limite_pct": 1},
        )

    # 5. Frecuencia deducida entre 100 y 1000 Hz
    fs = (n - 1) / float(tiempo[-1] - tiempo[0])
    if not (FS_MIN_HZ <= fs <= FS_MAX_HZ):
        raise ErrorAPI(
            "FRECUENCIA_FUERA_DE_RANGO",
            f"La frecuencia de muestreo deducida es {fs:.1f} Hz y debe estar entre 100 y 1000 Hz.",
            {"fs_hz": round(fs, 1), "minimo_hz": 100, "maximo_hz": 1000},
        )

    # 6. Duración entre 10 s y 300 s
    # Se redondea a centésimas: los tiempos con 6 decimales desvían fs una millonésima y un
    # archivo de exactamente 10 s saldría como 9.99999975 s.
    duracion = round(n / fs, 2)
    detalle_duracion = {
        "duracion_s": round(duracion, 1),
        "minimo_s": DURACION_MIN_S,
        "maximo_s": DURACION_MAX_S,
    }
    if duracion < DURACION_MIN_S:
        raise ErrorAPI(
            "DURACION_INSUFICIENTE",
            f"La señal dura {duracion:.1f} s y el mínimo es 10 s.",
            detalle_duracion,
        )
    if duracion > DURACION_MAX_S:
        raise ErrorAPI(
            "DURACION_EXCESIVA",
            f"La señal dura {duracion:.1f} s y el máximo es 300 s (5 min).",
            detalle_duracion,
        )

    # 7. Amplitud dentro de ±10 mV
    pico = float(np.max(np.abs(mv)))
    if pico > AMPLITUD_MAX_MV:
        raise ErrorAPI(
            "AMPLITUD_FUERA_DE_RANGO",
            f"La amplitud llega a {pico:.1f} mV y el máximo es 10 mV. ¿Está en milivoltios?",
            {"amplitud_max_mv": round(pico, 2), "limite_mv": AMPLITUD_MAX_MV},
        )

    return SenalLeida(tiempo_s=tiempo, mv=mv, fs_hz=fs, duracion_s=duracion)