import numpy as np
import pytest
import torch

from qcnn_ecg.config import FS, LONGITUD_VENTANA
from qcnn_ecg.entrenamiento import ConfigEntrenamiento, entrenar_modelo, fijar_semillas
from qcnn_ecg.evaluacion import predecir_probabilidades
from qcnn_ecg.modelos import CNN1D, ConfigCNN, Ensamble
from qcnn_ecg.pipeline import procesar_senal, rasgos_rr


def test_rasgos_rr_valores_conocidos():
    # RR de 1, 1, 2 y 1 s -> media 1.25
    picos = np.cumsum([0, 1, 1, 2, 1]) * FS
    r = rasgos_rr(picos)
    assert r.shape == (5, 4) and r.dtype == np.float32
    np.testing.assert_allclose(r[:, 0], [1, 0.8, 0.8, 1.6, 0.8])  # previo (primero: 1.0)
    np.testing.assert_allclose(r[:, 1], [0.8, 0.8, 1.6, 0.8, 1])  # posterior (último: 1.0)
    np.testing.assert_allclose(r[3, 2], np.mean([0.8, 0.8, 1.6]))  # local: previos hasta el 3
    np.testing.assert_allclose(r[3, 3], 1.6 / 0.8)  # previo/posterior
    assert r[0, 3] == 1 and r[-1, 3] == 1  # bordes sin vecino


def test_rasgos_rr_pocos_latidos():
    assert rasgos_rr(np.array([100])).shape == (1, 4)
    assert rasgos_rr(np.array([], dtype=int)).shape == (0, 4)


def test_procesar_senal_con_picos_dados():
    rng = np.random.default_rng(0)
    senal = rng.normal(0, 1, 10 * FS)
    picos = np.array([10, 500, 900, 1500, 3000, 3590])  # el primero y el último quedan en el borde
    lat = procesar_senal(senal, FS, picos=picos)
    assert lat.ventanas.shape == (4, LONGITUD_VENTANA) and lat.rr.shape == (4, 4)
    assert lat.mascara.tolist() == [False, True, True, True, True, False]
    assert not lat.remuestreada


def test_procesar_senal_remuestrea_y_detecta_con_detector_inyectado():
    senal = np.random.default_rng(1).normal(0, 1, 10 * 250)  # 250 Hz -> 2500 muestras a 360 Hz: 3600
    lat = procesar_senal(senal, 250, detector=lambda x: np.array([400, 1000, 2000]))
    assert lat.remuestreada and len(lat.picos) == 3 and lat.ventanas.shape[0] == 3


def test_cnn_con_rr_forma_y_error_sin_rr():
    modelo = CNN1D(ConfigCNN(rasgos_extra=4))
    assert modelo(torch.zeros(5, 1, 288), torch.ones(5, 4)).shape == (5,)
    with pytest.raises(ValueError):
        modelo(torch.zeros(5, 1, 288))


def test_cnn_sin_rr_ignora_rr_al_predecir():
    X = np.zeros((6, 288), dtype=np.float32)
    p = predecir_probabilidades(CNN1D(), X, rr=np.ones((6, 4), dtype=np.float32))
    assert p.shape == (6,)


def test_ensamble_promedia_probabilidades():
    a, b = CNN1D(), CNN1D()
    x = torch.randn(7, 1, 288)
    esperado = (torch.sigmoid(a.eval()(x)) + torch.sigmoid(b.eval()(x))) / 2
    ens = Ensamble([a, b]).eval()
    np.testing.assert_allclose(torch.sigmoid(ens(x)).detach(), esperado.detach(), atol=1e-5)


def test_media_centrada_deja_sin_puntuar_primera_y_ultima_epoca():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300).astype(np.int8)
    X = rng.normal(0, 1, (300, 288)).astype(np.float32)
    X[y == 1, 130:160] += 3.0
    rr = rng.normal(1, 0.1, (300, 4)).astype(np.float32)
    fijar_semillas(0)
    cfg = ConfigEntrenamiento(epocas_max=4, paciencia=10, tamano_lote=64, hilos=2)
    modelo, historial = entrenar_modelo(
        CNN1D(ConfigCNN(rasgos_extra=4)),
        X[:200], y[:200], X[200:], y[200:], cfg, 0, lambda _: None, rr[:200], rr[200:],
    )
    s = [h["f1_val_suavizado"] for h in historial]
    assert len(s) == 4 and np.isnan(s[0]) and np.isnan(s[-1])
    assert s[1] == pytest.approx(np.mean([h["f1_val"] for h in historial[:3]]))
    assert s[2] == pytest.approx(np.mean([h["f1_val"] for h in historial[1:4]]))
