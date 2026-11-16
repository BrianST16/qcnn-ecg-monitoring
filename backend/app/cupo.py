import threading
from contextlib import contextmanager

from app.errores import ErrorAPI


class Cupo:
    """Tope de análisis simultáneos: si no hay cupo libre, rechaza en vez de hacer cola."""

    def __init__(self, maximo: int):
        self._semaforo = threading.BoundedSemaphore(maximo)

    @contextmanager
    def reservar(self):
        if not self._semaforo.acquire(blocking=False):
            raise ErrorAPI(
                "SERVICIO_OCUPADO",
                "El servidor está atendiendo otros análisis. Inténtalo de nuevo en unos segundos.",
            )
        try:
            yield
        finally:
            self._semaforo.release()