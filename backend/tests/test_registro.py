import json
import time
from types import SimpleNamespace

import pytest
import torch
from qcnn_ecg.modelos.qcnn import QCNN, ConfigQCNN

from app.analisis import MENSAJE_QCNN, armar_resultados, armar_respuesta
from app.modelos.registro import cargar_modelos
from tests.test_analisis import construir_procesada, construir_senal


def ajustes(qcnn_modo="desactivada"):
    return SimpleNamespace(modelo_simulado=True, qcnn_modo=qcnn_modo, ruta_modelo_qcnn="no_importa.pt")


def test_qcnn_desactivada_solo_carga_la_cnn():
    assert list(cargar_modelos(ajustes("desactivada"))) == ["cnn"]


def test_qcnn_maqueta_carga_las_dos():
    modelos = cargar_modelos(ajustes("maqueta"))
    assert list(modelos) == ["cnn", "qcnn"]
    assert modelos["qcnn"].clave == "qcnn"


def test_qcnn_real_carga_el_ensamble_y_cumple_el_contrato(tmp_path):
    cfg = ConfigQCNN(rasgos_extra=4)
    ruta = tmp_path / "qcnn.pt"
    torch.save(
        {"estados": [QCNN(cfg).state_dict() for _ in range(2)], "config_modelo": cfg.a_dict(), "umbral": 0.4},
        ruta,
    )
    circuito = {"parametros_cuanticos": 18, "ancho": 8, "profundidad": 50, "compuertas_1q": 101,
                "compuertas_2q": 53, "compuertas_2q_conectividad_lineal": 83}
    ficha = {"id": "qcnn-x", "entrenado_con": "DS1", "circuito": circuito,
             "metricas_prueba": {"f1": 0.6, "exactitud": 0.8, "auc_roc": 0.87, "pacientes": 21}}
    ruta.with_suffix(".json").write_text(json.dumps(ficha), encoding="utf-8")
    ajustes_real = ajustes("real")
    ajustes_real.ruta_modelo_qcnn = ruta
    qcnn = cargar_modelos(ajustes_real)["qcnn"]
    assert qcnn.clave == "qcnn" and qcnn.umbral == pytest.approx(0.4)
    assert qcnn.id == "qcnn-x" and qcnn.circuito == circuito and qcnn.metricas_prueba["pacientes"] == 21
    assert qcnn.parametros == 2 * sum(p.numel() for p in QCNN(cfg).parameters())
    procesada = construir_procesada(6)
    p = qcnn.probabilidades(procesada.ventanas, procesada.rr)
    assert p.shape == (6,) and ((p >= 0) & (p <= 1)).all()


def test_la_maqueta_cumple_lo_que_exige_el_contrato():
    qcnn = cargar_modelos(ajustes("maqueta"))["qcnn"]
    assert isinstance(qcnn.id, str) and qcnn.id
    assert isinstance(qcnn.parametros, int)
    assert 0 <= qcnn.umbral <= 1
    probabilidades = qcnn.probabilidades(construir_procesada(8).ventanas)
    assert probabilidades.shape == (8,)
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()


def test_con_dos_modelos_los_latidos_son_comparables():
    modelos = cargar_modelos(ajustes("maqueta"))
    procesada = construir_procesada(10)
    resultados = armar_resultados(["cnn", "qcnn"], modelos, procesada)
    assert resultados.cnn.estado == resultados.qcnn.estado == "ok"
    cnn, qcnn = resultados.cnn.latidos, resultados.qcnn.latidos
    assert [l.indice for l in cnn] == [l.indice for l in qcnn]
    assert [l.tiempo_s for l in cnn] == [l.tiempo_s for l in qcnn]
    assert resultados.qcnn.modelo.id == "qcnn-maqueta"


def test_respuesta_completa_con_dos_modelos_es_valida():
    modelos = cargar_modelos(ajustes("maqueta"))
    procesada = construir_procesada(10)
    inicio = time.perf_counter()
    resultados = armar_resultados(["cnn", "qcnn"], modelos, procesada)
    respuesta = armar_respuesta("ecg.csv", construir_senal(), procesada, resultados, inicio)
    assert respuesta.resultados.qcnn.resumen.latidos_normales >= 0


def test_con_la_qcnn_desactivada_se_ve_no_disponible():
    modelos = cargar_modelos(ajustes("desactivada"))
    resultados = armar_resultados(["cnn", "qcnn"], modelos, construir_procesada(10))
    assert resultados.cnn.estado == "ok"
    assert resultados.qcnn.estado == "no_disponible"
    assert resultados.qcnn.mensaje == MENSAJE_QCNN