import threading

import pytest

from app.cupo import Cupo
from app.errores import ErrorAPI


def test_rechaza_cuando_no_hay_cupo():
    cupo = Cupo(1)
    with cupo.reservar():
        with pytest.raises(ErrorAPI) as e:
            with cupo.reservar():
                pass
    assert e.value.codigo == "SERVICIO_OCUPADO"


def test_permite_hasta_el_maximo():
    cupo = Cupo(2)
    with cupo.reservar():
        with cupo.reservar():
            with pytest.raises(ErrorAPI):
                with cupo.reservar():
                    pass


def test_libera_al_terminar():
    cupo = Cupo(1)
    with cupo.reservar():
        pass
    with cupo.reservar():  # si no se hubiera liberado, aquí fallaría
        pass


def test_libera_aunque_el_analisis_falle():
    cupo = Cupo(1)
    with pytest.raises(ValueError):
        with cupo.reservar():
            raise ValueError("el análisis falló")
    with cupo.reservar():
        pass


def test_con_hilos_reales():
    # 5 peticiones a la vez con cupo para 2: dos entran y tres se rechazan.
    cupo = Cupo(2)
    resultados = []
    todos_intentaron = threading.Barrier(5, timeout=5)

    def peticion():
        try:
            with cupo.reservar():
                resultados.append("ok")
                todos_intentaron.wait()  # quien entró espera, con el cupo ocupado
        except ErrorAPI:
            resultados.append("ocupado")
            todos_intentaron.wait()

    hilos = [threading.Thread(target=peticion) for _ in range(5)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    assert sorted(resultados) == ["ocupado", "ocupado", "ocupado", "ok", "ok"]