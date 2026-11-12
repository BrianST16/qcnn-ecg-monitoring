import math
import threading
import time
from collections import deque

from app.errores import ErrorAPI


class LimitadorPorIP:
    """Limita cuántas peticiones puede hacer cada IP en una ventana de tiempo deslizante.

    Guarda, por IP, la hora de sus últimas peticiones aceptadas. Si ya hay `maximo`
    dentro de los últimos `ventana_s` segundos, rechaza con 429. Todo vive en memoria:
    al reiniciar el servidor los contadores se borran (no se guarda nada en disco).
    """

    def __init__(self, maximo: int, ventana_s: float = 60.0, reloj=time.monotonic, max_ips: int = 10_000):
        if maximo < 1:
            raise ValueError("maximo debe ser al menos 1")
        self._maximo = maximo
        self._ventana_s = ventana_s
        self._reloj = reloj  # se puede reemplazar en las pruebas por un reloj falso
        self._max_ips = max_ips
        self._registros: dict[str, deque[float]] = {}
        self._cerrojo = threading.Lock()

    @property
    def ips_registradas(self) -> int:
        return len(self._registros)

    def registrar(self, ip: str) -> None:
        """Cuenta una petición de `ip`. Lanza ErrorAPI 429 si se pasó del límite."""
        ahora = self._reloj()
        with self._cerrojo:
            cola = self._registros.setdefault(ip, deque())
            self._descartar_vencidas(cola, ahora)
            if len(cola) >= self._maximo:
                # La más antigua es la primera que saldrá de la ventana.
                espera = max(1, math.ceil(cola[0] + self._ventana_s - ahora))
                raise ErrorAPI(
                    "DEMASIADAS_PETICIONES",
                    "Demasiadas peticiones. Inténtalo de nuevo en unos segundos.",
                    {"reintentar_en_s": espera},
                )
            cola.append(ahora)  # las peticiones rechazadas NO se cuentan
            if len(self._registros) > self._max_ips:
                self._limpiar(ahora)

    def _descartar_vencidas(self, cola: deque, ahora: float) -> None:
        limite = ahora - self._ventana_s
        while cola and cola[0] <= limite:
            cola.popleft()

    def _limpiar(self, ahora: float) -> None:
        """Olvida las IPs sin peticiones recientes, para que la memoria no crezca sin fin."""
        for ip in list(self._registros):
            cola = self._registros[ip]
            self._descartar_vencidas(cola, ahora)
            if not cola:
                del self._registros[ip]