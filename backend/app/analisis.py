import logging
import time
import uuid
from datetime import datetime, timezone

import numpy as np
from qcnn_ecg.config import FS

from app.csv_ecg import SenalLeida
from app.errores import ErrorAPI
from app.esquemas import (
    Deteccion,
    InfoModelo,
    Latido,
    RespuestaAnalisis,
    ResultadoModelo,
    Resultados,
    Resumen,
    Senal,
)
from app.modelos.base import Clasificador
from app.procesamiento import SenalProcesada

log = logging.getLogger("uvicorn.error")

CLAVES_MODELOS = ("cnn", "qcnn")
MENSAJE_QCNN = "La QCNN no está activa en este servidor."


def _redondear_o_none(valor: float, decimales: int) -> float | None:
    return None if np.isnan(valor) else round(float(valor), decimales)


def interpretar_modelos(texto: str) -> list[str]:
    """Convierte el campo `modelos` («cnn,qcnn») en una lista de claves válidas, sin repetidos."""
    claves = [c.strip().lower() for c in texto.split(",") if c.strip()]
    if not claves or any(c not in CLAVES_MODELOS for c in claves):
        raise ErrorAPI(
            "SOLICITUD_INVALIDA",
            "El campo «modelos» debe ser «cnn», «qcnn» o ambos separados por coma.",
            {"recibido": texto[:50], "validos": list(CLAVES_MODELOS)},
        )
    return list(dict.fromkeys(claves))


def ejecutar_modelo(clasificador: Clasificador, procesada: SenalProcesada) -> ResultadoModelo:
    """Clasifica todos los latidos con un modelo y arma su bloque de la respuesta."""
    inicio = time.perf_counter()
    probabilidades = np.asarray(clasificador.probabilidades(procesada.ventanas, procesada.rr), dtype=float)
    tiempo_ms = (time.perf_counter() - inicio) * 1000  # solo la inferencia del modelo

    n = procesada.latidos_evaluados
    if probabilidades.shape != (n,):
        raise ValueError(f"El modelo devolvió forma {probabilidades.shape} y se esperaba ({n},).")

    es_anormal = probabilidades >= clasificador.umbral  # regla del contrato
    anormales = int(es_anormal.sum())
    latidos = [
        Latido(
            indice=i,
            tiempo_s=round(float(procesada.tiempos_s[i]), 3),
            probabilidad_anormal=round(float(probabilidades[i]), 4),
            clase="ANORMAL" if es_anormal[i] else "NORMAL",
            rr_previo_s=_redondear_o_none(procesada.rr_previo_s[i], 3),
        )
        for i in range(n)
    ]
    return ResultadoModelo(
        estado="ok",
        modelo=InfoModelo(
            id=clasificador.id,
            nombre=clasificador.nombre,
            parametros=clasificador.parametros,
            umbral=round(clasificador.umbral, 4),
        ),
        tiempo_inferencia_ms=round(tiempo_ms, 1),
        resumen=Resumen(
            latidos_normales=n - anormales,
            latidos_anormales=anormales,
            porcentaje_anormal=round(100 * anormales / n, 1),
        ),
        latidos=latidos,
    )


def armar_resultados(
    pedidos: list[str], cargados: dict[str, Clasificador], procesada: SenalProcesada
) -> Resultados:
    """Un bloque por cada clave del contrato (cnn y qcnn), siempre las dos."""
    bloques = {}
    for clave in CLAVES_MODELOS:
        if clave not in pedidos:
            bloques[clave] = ResultadoModelo(
                estado="no_disponible", mensaje="No se solicitó este modelo en la petición."
            )
        elif clave not in cargados:
            mensaje = (
                MENSAJE_QCNN if clave == "qcnn" else "Este modelo no está cargado en el servidor."
            )
            bloques[clave] = ResultadoModelo(estado="no_disponible", mensaje=mensaje)
        else:
            try:
                bloques[clave] = ejecutar_modelo(cargados[clave], procesada)
            except Exception as error:
                # Solo el tipo de error: el mensaje podría arrastrar datos del archivo.
                log.error("Falló el modelo %s: %s", clave, type(error).__name__)
                bloques[clave] = ResultadoModelo(
                    estado="error", mensaje="El modelo no pudo clasificar este archivo."
                )
    return Resultados(**bloques)


def armar_respuesta(
    nombre_archivo: str | None,
    senal: SenalLeida,
    procesada: SenalProcesada,
    resultados: Resultados,
    inicio: float,
) -> RespuestaAnalisis:
    """Ensambla la respuesta 200 completa (sección 3.2 del contrato)."""
    return RespuestaAnalisis(
        id_analisis=uuid.uuid4(),
        fecha=datetime.now(timezone.utc).replace(microsecond=0),
        senal=Senal(
            nombre_archivo=(nombre_archivo or "sin_nombre.csv")[:255],
            derivacion="MLII",
            muestras=senal.muestras,
            duracion_s=senal.duracion_s,
            fs_original_hz=round(senal.fs_hz, 2),
            fs_modelo_hz=FS,
            remuestreada=procesada.remuestreada,
        ),
        deteccion=Deteccion(
            metodo="XQRS (wfdb)",
            latidos_detectados=procesada.latidos_detectados,
            latidos_evaluados=procesada.latidos_evaluados,
            latidos_descartados_borde=procesada.latidos_descartados_borde,
            frecuencia_cardiaca_media_lpm=round(procesada.fc_media_lpm, 1),
        ),
        resultados=resultados,
        tiempo_total_ms=round((time.perf_counter() - inicio) * 1000, 1),
    )