import time
from uuid import UUID

import numpy as np
import pytest

from app.analisis import armar_resultados, armar_respuesta, ejecutar_modelo, interpretar_modelos
from app.csv_ecg import SenalLeida
from app.errores import ErrorAPI
from app.esquemas import RespuestaAnalisis
from app.modelos.simulado import CNNSimulada
from app.procesamiento import SenalProcesada


def construir_procesada(n=6):
    """Resultado de procesar_senal con n latidos evaluados (el primero sin RR previo)."""
    return SenalProcesada(
        ventanas=np.zeros((n, 288)),
        rr=np.ones((n, 4), dtype=np.float32),
        tiempos_s=np.arange(1, n + 1) * 0.8,
        rr_previo_s=np.array([np.nan] + [0.8] * (n - 1)),
        latidos_detectados=n + 1,
        fc_media_lpm=75.0,
        fs_original_hz=360.0,
        remuestreada=False,
    )


def construir_senal(n=3600):
    return SenalLeida(
        tiempo_s=np.arange(n) / 360, mv=np.zeros(n), fs_hz=360.0000089, duracion_s=n / 360
    )


class ClasificadorFijo:
    """Modelo falso que devuelve siempre las mismas probabilidades."""

    clave = "cnn"
    id = "cnn-de-prueba"
    nombre = "CNN de prueba"
    parametros = 123
    umbral = 0.5

    def __init__(self, probabilidades):
        self.valores = np.array(probabilidades)

    def probabilidades(self, ventanas, rr=None):
        return self.valores


class ClasificadorRoto(ClasificadorFijo):
    def probabilidades(self, ventanas, rr=None):
        raise RuntimeError("fallo interno con datos del paciente")


def test_interpretar_modelos():
    assert interpretar_modelos("cnn,qcnn") == ["cnn", "qcnn"]
    assert interpretar_modelos(" CNN , cnn ,QCNN") == ["cnn", "qcnn"]  # sin repetidos
    assert interpretar_modelos("cnn") == ["cnn"]


@pytest.mark.parametrize("texto", ["", " , ", "xyz", "cnn,xyz"])
def test_modelos_invalidos(texto):
    with pytest.raises(ErrorAPI) as e:
        interpretar_modelos(texto)
    assert e.value.codigo == "SOLICITUD_INVALIDA"


def test_ejecutar_modelo_resumen_y_clases():
    modelo = ClasificadorFijo([0.1, 0.9, 0.5, 0.6, 0.7, 0.2])  # umbral 0.5
    r = ejecutar_modelo(modelo, construir_procesada(6))
    assert r.estado == "ok"
    clases = [x.clase for x in r.latidos]
    assert clases == ["NORMAL", "ANORMAL", "ANORMAL", "ANORMAL", "ANORMAL", "NORMAL"]
    assert r.resumen.latidos_normales == 2
    assert r.resumen.latidos_anormales == 4  # el 0.5 cuenta como anormal: la regla es >= umbral
    assert r.resumen.porcentaje_anormal == 66.7
    assert [x.indice for x in r.latidos] == [0, 1, 2, 3, 4, 5]
    assert r.latidos[0].rr_previo_s is None and r.latidos[1].rr_previo_s == 0.8
    assert r.latidos[0].tiempo_s == 0.8
    assert r.modelo.id == "cnn-de-prueba" and r.modelo.umbral == 0.5
    assert r.tiempo_inferencia_ms >= 0


def test_modelo_con_forma_incorrecta():
    with pytest.raises(ValueError):
        ejecutar_modelo(ClasificadorFijo([0.1, 0.2, 0.3]), construir_procesada(6))


def test_el_modelo_simulado_encaja():
    r = ejecutar_modelo(CNNSimulada(), construir_procesada(51))
    assert r.estado == "ok" and len(r.latidos) == 51
    assert all(0 <= x.probabilidad_anormal <= 1 for x in r.latidos)
    assert r.resumen.latidos_normales + r.resumen.latidos_anormales == 51
    assert r.modelo.id == "cnn1d-simulada"


def test_si_solo_se_pide_la_cnn():
    r = armar_resultados(["cnn"], {"cnn": ClasificadorFijo([0.1] * 6)}, construir_procesada(6))
    assert r.cnn.estado == "ok"
    assert r.qcnn.estado == "no_disponible"
    assert "No se solicitó" in r.qcnn.mensaje


def test_qcnn_no_disponible_con_el_mensaje_del_contrato():
    cargados = {"cnn": ClasificadorFijo([0.1] * 6)}
    r = armar_resultados(["cnn", "qcnn"], cargados, construir_procesada(6))
    assert r.qcnn.estado == "no_disponible"
    assert r.qcnn.mensaje == "La QCNN no está activa en este servidor."
    assert r.qcnn.modelo is None and r.qcnn.latidos is None


def test_un_modelo_roto_no_tumba_la_respuesta():
    cargados = {"cnn": ClasificadorRoto([0.1] * 6)}
    r = armar_resultados(["cnn", "qcnn"], cargados, construir_procesada(6))
    assert r.cnn.estado == "error"
    assert "paciente" not in r.cnn.mensaje  # el mensaje al cliente no filtra detalles internos
    assert r.qcnn.estado == "no_disponible"


def test_la_respuesta_tiene_la_forma_del_contrato():
    procesada = construir_procesada(6)
    resultados = armar_resultados(["cnn", "qcnn"], {"cnn": ClasificadorFijo([0.1] * 6)}, procesada)
    r = armar_respuesta("ecg.csv", construir_senal(), procesada, resultados, time.perf_counter())
    d = r.model_dump(mode="json")

    assert set(d) == {
        "id_analisis", "fecha", "advertencia", "senal", "deteccion",
        "resultados", "tiempo_total_ms",
    }
    assert set(d["senal"]) == {
        "nombre_archivo", "derivacion", "muestras", "duracion_s",
        "fs_original_hz", "fs_modelo_hz", "remuestreada",
    }
    assert set(d["deteccion"]) == {
        "metodo", "latidos_detectados", "latidos_evaluados",
        "latidos_descartados_borde", "frecuencia_cardiaca_media_lpm",
    }
    assert set(d["resultados"]) == {"cnn", "qcnn"}
    assert set(d["resultados"]["cnn"]["latidos"][0]) == {
        "indice", "tiempo_s", "probabilidad_anormal", "clase", "rr_previo_s",
    }
    UUID(d["id_analisis"])  # es un UUID válido
    assert d["fecha"].endswith("Z")  # UTC, como en el ejemplo
    assert d["senal"]["fs_original_hz"] == 360.0  # 360.0000089 se redondea
    assert d["senal"]["fs_modelo_hz"] == 360
    assert d["deteccion"]["latidos_descartados_borde"] == 1
    assert d["advertencia"].startswith("Resultado experimental")
    RespuestaAnalisis.model_validate_json(r.model_dump_json())  # ida y vuelta sin errores


def test_nombre_de_archivo_ausente_o_demasiado_largo():
    procesada = construir_procesada(6)
    resultados = armar_resultados(["cnn"], {"cnn": ClasificadorFijo([0.1] * 6)}, procesada)
    senal = construir_senal()
    sin_nombre = armar_respuesta(None, senal, procesada, resultados, time.perf_counter())
    largo = armar_respuesta("a" * 300, senal, procesada, resultados, time.perf_counter())
    assert sin_nombre.senal.nombre_archivo == "sin_nombre.csv"
    assert len(largo.senal.nombre_archivo) == 255