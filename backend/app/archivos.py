from fastapi import UploadFile

from app.errores import ErrorAPI

MAX_BYTES = 5 * 1024 * 1024  # 5 MB (sección 2 del contrato)


def leer_texto_csv(archivo: UploadFile) -> str:
    """Lee el archivo subido y devuelve su texto. Lanza ErrorAPI si no cumple el contrato."""
    contenido = archivo.file.read(MAX_BYTES + 1)  # +1 para detectar que se pasó del máximo

    if len(contenido) == 0:
        raise ErrorAPI("ARCHIVO_VACIO", "El archivo está vacío.")
    if len(contenido) > MAX_BYTES:
        raise ErrorAPI(
            "ARCHIVO_DEMASIADO_GRANDE",
            "El archivo supera el máximo de 5 MB.",
            {"maximo_mb": 5},
        )

    nombre = (archivo.filename or "").lower()
    if not nombre.endswith(".csv"):
        raise ErrorAPI(
            "TIPO_NO_SOPORTADO",
            "El archivo debe tener extensión .csv.",
            {"nombre_archivo": archivo.filename},
        )

    try:
        return contenido.decode("utf-8-sig")  # acepta UTF-8 con o sin BOM
    except UnicodeDecodeError:
        raise ErrorAPI(
            "TIPO_NO_SOPORTADO", "El archivo no es un texto UTF-8 válido."
        ) from None