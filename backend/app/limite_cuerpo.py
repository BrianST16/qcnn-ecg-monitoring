from starlette.responses import JSONResponse

HOLGURA_MULTIPART = 64 * 1024  # el formato multipart añade encabezados y delimitadores al archivo


class LimiteTamanoCuerpo:
    """Rechaza (413) las peticiones con cuerpo demasiado grande antes de que la app las procese.

    1. Si el cliente declara `Content-Length` mayor que el límite, responde sin leer nada.
    2. Si no lo declara (envío por trozos), cuenta los bytes a medida que llegan y corta al pasarse.
    """

    def __init__(self, app, maximo_bytes: int, maximo_mb: int):
        self.app = app
        self.maximo_bytes = maximo_bytes
        self.maximo_mb = maximo_mb

    async def _rechazar(self, scope, receive, send):
        respuesta = JSONResponse(
            status_code=413,
            content={
                "error": {
                    "codigo": "ARCHIVO_DEMASIADO_GRANDE",
                    "mensaje": f"La petición supera el máximo de {self.maximo_mb} MB.",
                    "detalle": {"maximo_mb": self.maximo_mb},
                }
            },
        )
        await respuesta(scope, receive, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declarado = dict(scope["headers"]).get(b"content-length")
        if declarado is not None and declarado.isdigit() and int(declarado) > self.maximo_bytes:
            await self._rechazar(scope, receive, send)
            return

        recibido = 0
        rechazada = False

        async def receive_vigilado():
            nonlocal recibido, rechazada
            if rechazada:
                return {"type": "http.disconnect"}
            mensaje = await receive()
            if mensaje["type"] == "http.request":
                recibido += len(mensaje.get("body", b""))
                if recibido > self.maximo_bytes:
                    rechazada = True
                    await self._rechazar(scope, receive, send)
                    return {"type": "http.disconnect"}  # a la app le decimos que el cliente se fue
            return mensaje

        async def send_vigilado(mensaje):
            if not rechazada:  # tras rechazar, la respuesta de la app (un error de lectura) sobra
                await send(mensaje)

        await self.app(scope, receive_vigilado, send_vigilado)