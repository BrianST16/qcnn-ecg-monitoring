import secrets

from fastapi import Depends
from fastapi.security import APIKeyHeader

from app.config import Ajustes, get_ajustes
from app.errores import ErrorAPI

esquema_api_key = APIKeyHeader(name="X-API-Key", auto_error=False)


def exigir_api_key(
    clave: str | None = Depends(esquema_api_key),
    ajustes: Ajustes = Depends(get_ajustes),
) -> None:
    valida = clave is not None and secrets.compare_digest(
        clave.encode("utf-8"), ajustes.api_key.encode("utf-8")
    )
    if not valida:
        raise ErrorAPI("NO_AUTORIZADO", "Falta la clave de acceso o no es válida.")
    