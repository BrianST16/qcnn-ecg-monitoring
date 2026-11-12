from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

ADVERTENCIA = "Resultado experimental con fines académicos. No constituye un diagnóstico médico."


# ---------- GET /salud ----------
class RespuestaSalud(BaseModel):
    estado: str
    modelos_cargados: list[str]
    version_api: str


# ---------- GET /modelos ----------
class MetricasPrueba(BaseModel):
    f1: float
    exactitud: float
    auc_roc: float
    pacientes: int


class FichaModelo(BaseModel):
    id: str | None
    clave: Literal["cnn", "qcnn"]
    nombre: str
    disponible: bool
    parametros: int | None
    umbral: float | None
    entrenado_con: str | None
    metricas_prueba: MetricasPrueba | None


class RespuestaModelos(BaseModel):
    modelos: list[FichaModelo]


# ---------- POST /analisis (respuesta 200) ----------
class Senal(BaseModel):
    nombre_archivo: str
    derivacion: str
    muestras: int
    duracion_s: float
    fs_original_hz: float
    fs_modelo_hz: int
    remuestreada: bool


class Deteccion(BaseModel):
    metodo: str
    latidos_detectados: int
    latidos_evaluados: int
    latidos_descartados_borde: int
    frecuencia_cardiaca_media_lpm: float


class InfoModelo(BaseModel):
    id: str
    nombre: str
    parametros: int
    umbral: float


class Resumen(BaseModel):
    latidos_normales: int
    latidos_anormales: int
    porcentaje_anormal: float


class Latido(BaseModel):
    indice: int
    tiempo_s: float
    probabilidad_anormal: float = Field(ge=0, le=1)
    clase: Literal["NORMAL", "ANORMAL"]
    rr_previo_s: float | None


class ResultadoModelo(BaseModel):
    estado: Literal["ok", "no_disponible", "error"]
    mensaje: str | None = None
    modelo: InfoModelo | None = None
    tiempo_inferencia_ms: float | None = None
    resumen: Resumen | None = None
    latidos: list[Latido] | None = None


class Resultados(BaseModel):
    cnn: ResultadoModelo
    qcnn: ResultadoModelo


class RespuestaAnalisis(BaseModel):
    id_analisis: UUID
    fecha: datetime
    advertencia: str = ADVERTENCIA
    senal: Senal
    deteccion: Deteccion
    resultados: Resultados
    tiempo_total_ms: float
    
class CuerpoError(BaseModel):
    codigo: str
    mensaje: str
    detalle: dict = Field(default_factory=dict)


class RespuestaError(BaseModel):
    error: CuerpoError