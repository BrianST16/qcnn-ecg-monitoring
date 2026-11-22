import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile

from app.analisis import armar_resultados, armar_respuesta, interpretar_modelos
from app.archivos import leer_texto_csv, MAX_BYTES
from app.config import get_ajustes
from app.csv_ecg import leer_senal
from app.cupo import Cupo
from app.errores import registrar_manejadores
from app.esquemas import FichaModelo, RespuestaAnalisis, RespuestaModelos, RespuestaSalud
from app.modelos.registro import cargar_modelos
from app.procesamiento import procesar_senal
from app.seguridad import exigir_api_key

from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.limite_cuerpo import HOLGURA_MULTIPART, LimiteTamanoCuerpo

from app.limitador import LimitadorPorIP

log = logging.getLogger("uvicorn.error")
ajustes = get_ajustes()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.modelos = cargar_modelos(ajustes)  # una vez, al iniciar
    app.state.cupo = Cupo(ajustes.max_analisis_simultaneos)
    app.state.limitador = LimitadorPorIP(ajustes.limite_peticiones_por_minuto)
    yield  # aquí la app atiende peticiones
    app.state.modelos.clear()  # al apagar


app = FastAPI(
    title="API de análisis de ECG (CNN y QCNN)",
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs" if ajustes.docs_habilitados else None,
    redoc_url="/redoc" if ajustes.docs_habilitados else None,
    openapi_url="/openapi.json" if ajustes.docs_habilitados else None,
)
registrar_manejadores(app, trazas=ajustes.docs_habilitados)

# El último middleware que se añade es el más externo: el primero en ver la petición.
# Orden de entrada: hosts -> CORS -> tamaño del cuerpo -> app.
app.add_middleware(
    LimiteTamanoCuerpo,
    maximo_bytes=MAX_BYTES + HOLGURA_MULTIPART,
    maximo_mb=MAX_BYTES // (1024 * 1024),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ajustes.lista_origenes,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ajustes.lista_hosts)

def limitar_peticiones(request: Request) -> None:
    ip = request.client.host if request.client else "desconocida"
    request.app.state.limitador.registrar(ip)

@app.get("/api/v1/salud", response_model=RespuestaSalud)
def salud(request: Request):
    modelos = request.app.state.modelos
    return {"estado": "ok", "modelos_cargados": list(modelos), "version_api": "1.1.0"}


MODELOS_CONOCIDOS = (("cnn", "CNN 1-D"), ("qcnn", "QCNN (simulada)"))


@app.get(
    "/api/v1/modelos",
    response_model=RespuestaModelos,
    dependencies=[Depends(limitar_peticiones), Depends(exigir_api_key)]
)
def listar_modelos(request: Request):
    cargados = request.app.state.modelos
    fichas = []
    for clave, nombre in MODELOS_CONOCIDOS:
        m = cargados.get(clave)
        if m is None:
            fichas.append(
                FichaModelo(
                    id=None, clave=clave, nombre=nombre, disponible=False,
                    parametros=None, umbral=None, entrenado_con=None, metricas_prueba=None,
                )
            )
        else:
            fichas.append(
                FichaModelo(
                    id=m.id, clave=clave, nombre=m.nombre, disponible=True,
                    parametros=m.parametros, umbral=round(m.umbral, 4),
                    entrenado_con=m.entrenado_con, metricas_prueba=m.metricas_prueba,
                    circuito=m.circuito,
                )
            )
    return RespuestaModelos(modelos=fichas)


@app.post(
    "/api/v1/analisis",
    response_model=RespuestaAnalisis,
    dependencies=[Depends(limitar_peticiones), Depends(exigir_api_key)]
)
def analizar(
    request: Request,
    archivo: UploadFile = File(...),
    modelos: str = Form("cnn,qcnn"),
):
    inicio = time.perf_counter()
    pedidos = interpretar_modelos(modelos)  # barato: si es inválido, falla antes de gastar cupo
    with request.app.state.cupo.reservar():
        texto = leer_texto_csv(archivo)
        senal = leer_senal(texto)
        procesada = procesar_senal(senal)
        resultados = armar_resultados(pedidos, request.app.state.modelos, procesada)
        respuesta = armar_respuesta(archivo.filename, senal, procesada, resultados, inicio)
    # Log limpio: nada del archivo ni del paciente, solo cifras del análisis.
    log.info(
        "Análisis %s: %d latidos evaluados en %.0f ms",
        respuesta.id_analisis,
        respuesta.deteccion.latidos_evaluados,
        respuesta.tiempo_total_ms,
    )
    return respuesta