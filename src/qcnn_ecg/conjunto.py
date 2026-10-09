"""Construcción, caché versionada y carga del conjunto de latidos.

El conjunto procesado se guarda en data/processed/<nombre>/ (fuera de git) y se
describe en data/manifiestos/<nombre>.json (versionado) con parámetros, registros
por partición, conteos y SHA-256 de cada archivo. Regenerarlo debe dar los mismos
hashes.
"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from qcnn_ecg import config
from qcnn_ecg.datos import leer_registro, registros_estudio
from qcnn_ecg.particion import asignar_particiones, elegir_validacion
from qcnn_ecg.preprocesamiento import etiquetar_simbolo, extraer_ventanas, preprocesar_senal

ARCHIVOS = ("X", "y", "registro", "simbolo", "muestra_r", "particion")


@dataclass
class Conjunto:
    """Latidos segmentados con sus metadatos (arrays alineados por latido)."""

    X: np.ndarray  # (n, 288) float32
    y: np.ndarray  # (n,) int8
    registro: np.ndarray  # (n,) str
    simbolo: np.ndarray  # (n,) str
    muestra_r: np.ndarray  # (n,) int64, posición del pico R en el registro
    particion: np.ndarray  # (n,) str

    def mascara(self, particion: str) -> np.ndarray:
        return self.particion == particion

    def subconjunto(self, particion: str) -> tuple[np.ndarray, np.ndarray]:
        m = self.mascara(particion)
        return self.X[m], self.y[m]


def procesar_registros(nombres, directorio: Path = config.DIR_MITDB) -> dict[str, np.ndarray]:
    """Preprocesa y segmenta los registros dados (sin asignar partición)."""
    partes = {k: [] for k in ("X", "y", "registro", "simbolo", "muestra_r")}
    for nombre in nombres:
        reg = leer_registro(nombre, directorio)
        senal = preprocesar_senal(reg.senal, reg.fs)
        etiquetas = np.array([etiquetar_simbolo(s) for s in reg.simbolos], dtype=object)
        usar = etiquetas != None  # noqa: E711 (comparación elemento a elemento)
        ventanas, dentro = extraer_ventanas(senal, reg.muestras[usar])
        partes["X"].append(ventanas.astype(np.float32))
        partes["y"].append(etiquetas[usar][dentro].astype(np.int8))
        partes["simbolo"].append(reg.simbolos[usar][dentro].astype("<U1"))
        partes["muestra_r"].append(reg.muestras[usar][dentro])
        partes["registro"].append(np.full(len(ventanas), nombre, dtype="<U3"))
    return {k: np.concatenate(v) for k, v in partes.items()}


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def guardar_arrays(arrays: dict[str, np.ndarray], directorio: Path) -> dict[str, str]:
    """Guarda cada array como .npy y devuelve su SHA-256."""
    directorio.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for nombre, arr in arrays.items():
        ruta = directorio / f"{nombre}.npy"
        np.save(ruta, arr, allow_pickle=False)
        hashes[nombre] = _sha256(ruta)
    return hashes


def _manifiesto(arrays: dict[str, np.ndarray], validacion, hashes) -> dict:
    registros_por_particion = {
        p: sorted(set(arrays["registro"][arrays["particion"] == p].tolist()))
        for p in config.PARTICIONES
    }
    conteos = {}
    for p in config.PARTICIONES:
        m = arrays["particion"] == p
        conteos[p] = {
            "latidos": int(m.sum()),
            "normal": int((arrays["y"][m] == 0).sum()),
            "anormal": int((arrays["y"][m] == 1).sum()),
            "simbolos": dict(sorted(Counter(arrays["simbolo"][m].tolist()).items())),
        }
    return {
        "nombre": config.NOMBRE_CONJUNTO,
        "base": "MIT-BIH Arrhythmia Database (PhysioNet, mitdb 1.0.0)",
        "parametros": {
            "fs_hz": config.FS,
            "derivacion": config.DERIVACION,
            "banda_hz": list(config.BANDA_HZ),
            "orden_filtro": config.ORDEN_FILTRO,
            "filtro": "Butterworth pasa-banda, filtfilt (fase cero)",
            "normalizacion": "z-score por registro",
            "media_ventana": config.MEDIA_VENTANA,
            "longitud_ventana": config.LONGITUD_VENTANA,
            "simbolos_normal": list(config.SIMBOLOS_NORMAL),
            "simbolos_anormal": list(config.SIMBOLOS_ANORMAL),
            "excluidos_marcapasos": list(config.MARCAPASOS),
            "excluidos_prueba": list(config.EXCLUIDOS_PRUEBA),
            "semilla_particion": config.SEMILLA_PARTICION,
        },
        "registros_validacion": list(validacion),
        "registros_por_particion": registros_por_particion,
        "conteos": conteos,
        "forma_X": list(arrays["X"].shape),
        "sha256": hashes,
        "versiones": {p: version(p) for p in ("numpy", "scipy", "wfdb", "scikit-learn")},
    }


def construir_conjunto(
    directorio_mitdb: Path = config.DIR_MITDB,
    directorio_procesados: Path = config.DIR_PROCESADOS,
    directorio_manifiestos: Path = config.DIR_MANIFIESTOS,
) -> dict:
    """Construye el conjunto completo, lo guarda y escribe el manifiesto."""
    arrays = procesar_registros(registros_estudio(), directorio_mitdb)

    en_ds1 = np.isin(arrays["registro"], config.DS1)
    validacion = elegir_validacion(
        arrays["registro"][en_ds1], arrays["y"][en_ds1], arrays["simbolo"][en_ds1]
    )
    asignacion = asignar_particiones(validacion)
    arrays["particion"] = np.array([asignacion[r] for r in arrays["registro"]], dtype="<U15")

    hashes = guardar_arrays(arrays, Path(directorio_procesados) / config.NOMBRE_CONJUNTO)
    manifiesto = _manifiesto(arrays, validacion, hashes)
    directorio_manifiestos = Path(directorio_manifiestos)
    directorio_manifiestos.mkdir(parents=True, exist_ok=True)
    ruta = directorio_manifiestos / f"{config.NOMBRE_CONJUNTO}.json"
    ruta.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifiesto


def cargar_conjunto(
    directorio_procesados: Path = config.DIR_PROCESADOS,
    directorio_manifiestos: Path = config.DIR_MANIFIESTOS,
    verificar: bool = True,
) -> Conjunto:
    """Carga el conjunto procesado; con `verificar` comprueba los SHA del manifiesto."""
    directorio = Path(directorio_procesados) / config.NOMBRE_CONJUNTO
    if not directorio.exists():
        raise FileNotFoundError(
            f"No existe {directorio}. Ejecuta: uv run python -m qcnn_ecg.preparar_datos"
        )
    if verificar:
        ruta_manifiesto = Path(directorio_manifiestos) / f"{config.NOMBRE_CONJUNTO}.json"
        esperado = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))["sha256"]
        for nombre in ARCHIVOS:
            if _sha256(directorio / f"{nombre}.npy") != esperado[nombre]:
                raise ValueError(f"{nombre}.npy no coincide con el manifiesto; regenéralo")
    return Conjunto(**{n: np.load(directorio / f"{n}.npy") for n in ARCHIVOS})


class CargadorLotes:
    """Iterador de lotes sobre tensores en memoria (más rápido que DataLoader aquí)."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tamano_lote: int,
        barajar: bool = False,
        semilla: int = 0,
    ):
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).unsqueeze(1)
        self.y = torch.from_numpy(y.astype(np.float32))
        self.tamano_lote = tamano_lote
        self.barajar = barajar
        self.generador = torch.Generator().manual_seed(semilla)

    def __len__(self) -> int:
        return (len(self.y) + self.tamano_lote - 1) // self.tamano_lote

    def __iter__(self):
        n = len(self.y)
        orden = torch.randperm(n, generator=self.generador) if self.barajar else torch.arange(n)
        for i in range(0, n, self.tamano_lote):
            idx = orden[i : i + self.tamano_lote]
            yield self.X[idx], self.y[idx]
