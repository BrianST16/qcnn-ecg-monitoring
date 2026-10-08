"""Lectura de la base MIT-BIH Arrhythmia desde disco (formato WFDB)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

from qcnn_ecg.config import DERIVACION, DIR_MITDB, DS1, DS2, FS


@dataclass(frozen=True)
class Registro:
    """Un registro de MIT-BIH reducido a una derivación y sus anotaciones."""

    nombre: str
    senal: np.ndarray  # (n_muestras,) en mV
    fs: int
    muestras: np.ndarray  # posición (en muestras) de cada anotación
    simbolos: np.ndarray  # símbolo de cada anotación ('N', 'V', '+', ...)
    derivacion: str
    indice_canal: int


def registros_estudio() -> tuple[str, ...]:
    """Registros usados en el estudio: DS1 + DS2 (sin los de marcapasos)."""
    return DS1 + DS2


def descargar_mitdb(directorio: Path = DIR_MITDB) -> Path:
    """Descarga MIT-BIH desde PhysioNet si faltan registros en `directorio`."""
    directorio = Path(directorio)
    faltantes = [
        r
        for r in registros_estudio()
        if not all((directorio / f"{r}.{ext}").exists() for ext in ("hea", "dat", "atr"))
    ]
    if faltantes:
        directorio.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database("mitdb", str(directorio), records=faltantes)
    return directorio


def leer_anotaciones(nombre: str, directorio: Path = DIR_MITDB) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (muestras, simbolos) de las anotaciones de referencia (.atr)."""
    anotacion = wfdb.rdann(str(Path(directorio) / nombre), "atr")
    return np.asarray(anotacion.sample, dtype=np.int64), np.asarray(anotacion.symbol)


def leer_registro(
    nombre: str, directorio: Path = DIR_MITDB, derivacion: str = DERIVACION
) -> Registro:
    """Lee la derivación pedida (por nombre, no por posición) y sus anotaciones."""
    ruta = str(Path(directorio) / nombre)
    encabezado = wfdb.rdheader(ruta)
    if derivacion not in encabezado.sig_name:
        raise ValueError(f"El registro {nombre} no tiene la derivación {derivacion}")
    if encabezado.fs != FS:
        raise ValueError(f"El registro {nombre} tiene fs={encabezado.fs}, se esperaba {FS}")

    indice = encabezado.sig_name.index(derivacion)
    registro = wfdb.rdrecord(ruta, channels=[indice])
    muestras, simbolos = leer_anotaciones(nombre, directorio)
    return Registro(
        nombre=nombre,
        senal=registro.p_signal[:, 0].astype(np.float64),
        fs=int(registro.fs),
        muestras=muestras,
        simbolos=simbolos,
        derivacion=derivacion,
        indice_canal=indice,
    )
