import threading

import pytest

from app.errores import ErrorAPI
from app.limitador import LimitadorPorIP


class RelojFalso:
    """Reloj que avanza solo cuando la prueba lo dice, para no esperar segundos reales."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def avanzar(self, segundos):
        self.t += segundos


def _crear(maximo=3, **extra):
    reloj = RelojFalso()
    return LimitadorPorIP(maximo, ventana_s=60, reloj=reloj, **extra), reloj


def test_permite_hasta_el_maximo():
    limitador, _ = _crear(3)
    for _ in range(3):
        limitador.registrar("1.1.1.1")


def test_rechaza_la_peticion_que_excede_el_maximo():
    limitador, _ = _crear(3)
    for _ in range(3):
        limitador.registrar("1.1.1.1")
    with pytest.raises(ErrorAPI) as info:
        limitador.registrar("1.1.1.1")
    assert info.value.codigo == "DEMASIADAS_PETICIONES"
    assert info.value.estado == 429
    assert info.value.detalle == {"reintentar_en_s": 60}


def test_cada_ip_tiene_su_propio_contador():
    limitador, _ = _crear(1)
    limitador.registrar("1.1.1.1")
    limitador.registrar("2.2.2.2")  # otra IP: no se ve afectada
    with pytest.raises(ErrorAPI):
        limitador.registrar("1.1.1.1")


def test_la_ventana_es_deslizante():
    limitador, reloj = _crear(2)
    limitador.registrar("ip")  # t = 0
    reloj.avanzar(30)
    limitador.registrar("ip")  # t = 30
    reloj.avanzar(10)  # t = 40: lleno
    with pytest.raises(ErrorAPI) as info:
        limitador.registrar("ip")
    assert info.value.detalle["reintentar_en_s"] == 20  # la de t=0 sale en t=60
    reloj.avanzar(20)  # t = 60: la primera ya salió, la segunda aún no
    limitador.registrar("ip")
    with pytest.raises(ErrorAPI):
        limitador.registrar("ip")


def test_las_peticiones_rechazadas_no_cuentan():
    limitador, reloj = _crear(2)
    limitador.registrar("ip")
    limitador.registrar("ip")
    for _ in range(5):  # insistir no debe alargar el castigo
        reloj.avanzar(10)
        with pytest.raises(ErrorAPI):
            limitador.registrar("ip")
    reloj.avanzar(10)  # t = 60
    limitador.registrar("ip")


def test_el_tiempo_de_espera_nunca_es_menor_a_un_segundo():
    limitador, reloj = _crear(1)
    limitador.registrar("ip")
    reloj.avanzar(59.9)
    with pytest.raises(ErrorAPI) as info:
        limitador.registrar("ip")
    assert info.value.detalle["reintentar_en_s"] == 1


def test_olvida_las_ips_inactivas_cuando_hay_demasiadas():
    limitador, reloj = _crear(5, max_ips=3)
    for i in range(4):
        limitador.registrar(f"10.0.0.{i}")
    assert limitador.ips_registradas == 4  # aún no vencen: no se puede borrar nada
    reloj.avanzar(61)
    limitador.registrar("10.0.1.0")  # al pasar de max_ips se limpian las vencidas
    assert limitador.ips_registradas == 1


def test_con_hilos_a_la_vez_nunca_pasa_mas_del_maximo():
    limitador, _ = _crear(5)
    barrera = threading.Barrier(12)
    resultados = []

    def intento():
        barrera.wait()
        try:
            limitador.registrar("ip")
            resultados.append("ok")
        except ErrorAPI:
            resultados.append("429")

    hilos = [threading.Thread(target=intento) for _ in range(12)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    assert resultados.count("ok") == 5
    assert resultados.count("429") == 7


def test_maximo_invalido_se_rechaza():
    with pytest.raises(ValueError):
        LimitadorPorIP(0)