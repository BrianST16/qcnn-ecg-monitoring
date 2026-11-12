import asyncio
import json

from app.limite_cuerpo import LimiteTamanoCuerpo

LIMITE = 1000  # bytes, para probar con cuerpos pequeños


async def app_de_prueba(scope, receive, send):
    """App mínima: lee todo el cuerpo y responde 200; si el cliente 'se desconecta', 400."""
    llamadas.append(1)
    while True:
        mensaje = await receive()
        if mensaje["type"] == "http.disconnect":
            estado = 400
            break
        if not mensaje.get("more_body", False):
            estado = 200
            break
    await send({"type": "http.response.start", "status": estado, "headers": []})
    await send({"type": "http.response.body", "body": b"app"})


llamadas = []


def enviar(trozos, content_length=None, tipo="http"):
    """Simula una petición y devuelve los mensajes que el middleware le manda al servidor."""
    llamadas.clear()
    cabeceras = [(b"host", b"localhost")]
    if content_length is not None:
        cabeceras.append((b"content-length", str(content_length).encode()))
    scope = {"type": tipo, "method": "POST", "path": "/analisis", "headers": cabeceras}
    pendientes = [
        {"type": "http.request", "body": t, "more_body": i < len(trozos) - 1}
        for i, t in enumerate(trozos)
    ]
    enviados = []

    async def receive():
        return pendientes.pop(0) if pendientes else {"type": "http.disconnect"}

    async def send(mensaje):
        enviados.append(mensaje)

    app = LimiteTamanoCuerpo(app_de_prueba, maximo_bytes=LIMITE, maximo_mb=5)
    asyncio.run(app(scope, receive, send))
    return enviados


def estado(enviados):
    inicios = [m for m in enviados if m["type"] == "http.response.start"]
    assert len(inicios) == 1, "debe haber exactamente una respuesta"
    return inicios[0]["status"]


def cuerpo(enviados):
    return json.loads(b"".join(m.get("body", b"") for m in enviados))


def test_deja_pasar_un_cuerpo_pequeno():
    enviados = enviar([b"x" * 500], content_length=500)
    assert estado(enviados) == 200
    assert llamadas == [1]


def test_content_length_demasiado_grande_se_rechaza_sin_leer():
    enviados = enviar([b"x" * 10], content_length=5000)
    assert estado(enviados) == 413
    assert llamadas == []  # la app ni se enteró
    error = cuerpo(enviados)["error"]
    assert error["codigo"] == "ARCHIVO_DEMASIADO_GRANDE"
    assert error["detalle"] == {"maximo_mb": 5}


def test_el_limite_exacto_pasa():
    enviados = enviar([b"x" * LIMITE], content_length=LIMITE)
    assert estado(enviados) == 200


def test_envio_por_trozos_dentro_del_limite():
    enviados = enviar([b"x" * 400, b"x" * 400])  # sin Content-Length
    assert estado(enviados) == 200


def test_envio_por_trozos_que_se_pasa_se_corta():
    # Sin Content-Length: el middleware cuenta los bytes y corta al pasar el límite.
    enviados = enviar([b"x" * 600, b"x" * 600, b"x" * 600])
    assert estado(enviados) == 413  # una sola respuesta: la 413, no el 400 de la app
    assert cuerpo(enviados)["error"]["codigo"] == "ARCHIVO_DEMASIADO_GRANDE"


def test_ignora_lo_que_no_es_http():
    enviados = enviar([], tipo="lifespan")
    assert llamadas == [1]  # pasó directo a la app, sin contar ni rechazar
    assert enviados[0]["type"] == "http.response.start"