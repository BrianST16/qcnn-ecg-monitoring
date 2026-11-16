import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.esquemas import CuerpoError, RespuestaError

log = logging.getLogger("uvicorn.error")

# Código de error -> estado HTTP (sección 3.3 del contrato y los tres códigos propuestos)
ESTADOS = {
    "ARCHIVO_VACIO": 400,
    "COLUMNAS_INVALIDAS": 400,
    "VALORES_INVALIDOS": 400,
    "TIEMPO_NO_CRECIENTE": 400,
    "MUESTREO_IRREGULAR": 400,
    "NO_AUTORIZADO": 401,
    "ARCHIVO_DEMASIADO_GRANDE": 413,
    "TIPO_NO_SOPORTADO": 415,
    "FRECUENCIA_FUERA_DE_RANGO": 422,
    "DURACION_INSUFICIENTE": 422,
    "DURACION_EXCESIVA": 422,
    "AMPLITUD_FUERA_DE_RANGO": 422,
    "SIN_LATIDOS_SUFICIENTES": 422,
    "SOLICITUD_INVALIDA": 422,
    "DEMASIADAS_PETICIONES": 429,
    "ERROR_INTERNO": 500,
    "SERVICIO_OCUPADO": 503,
}


class ErrorAPI(Exception):
    """Error esperado del contrato. Se lanza con `raise ErrorAPI(codigo, mensaje, detalle)`."""

    def __init__(self, codigo: str, mensaje: str, detalle: dict | None = None):
        self.codigo = codigo
        self.mensaje = mensaje
        self.detalle = detalle or {}
        self.estado = ESTADOS[codigo]
        super().__init__(mensaje)


def _respuesta(estado: int, codigo: str, mensaje: str, detalle: dict | None = None) -> JSONResponse:
    cuerpo = RespuestaError(error=CuerpoError(codigo=codigo, mensaje=mensaje, detalle=detalle or {}))
    return JSONResponse(status_code=estado, content=cuerpo.model_dump())


def registrar_manejadores(app: FastAPI, trazas: bool) -> None:
    @app.exception_handler(ErrorAPI)
    async def _error_api(request: Request, exc: ErrorAPI):
        return _respuesta(exc.estado, exc.codigo, exc.mensaje, exc.detalle)

    @app.exception_handler(RequestValidationError)
    async def _validacion(request: Request, exc: RequestValidationError):
        campos = [str(e["loc"][-1]) for e in exc.errors() if e.get("loc")]
        if "archivo" in campos:
            return _respuesta(400, "ARCHIVO_VACIO", "Falta el archivo CSV en el campo «archivo».")
        return _respuesta(
            422, "SOLICITUD_INVALIDA", "La solicitud no tiene el formato esperado.", {"campos": campos}
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        mensajes = {404: "La ruta no existe.", 405: "Método no permitido para esta ruta."}
        return _respuesta(
            exc.status_code, "SOLICITUD_INVALIDA", mensajes.get(exc.status_code, "Solicitud inválida.")
        )

    @app.exception_handler(Exception)
    async def _inesperado(request: Request, exc: Exception):
        if trazas:
            log.error("Error interno en %s %s", request.method, request.url.path, exc_info=exc)
        else:
            log.error("Error interno en %s %s: %s", request.method, request.url.path, type(exc).__name__)
        return _respuesta(500, "ERROR_INTERNO", "Ocurrió un error interno. Inténtalo de nuevo más tarde.")