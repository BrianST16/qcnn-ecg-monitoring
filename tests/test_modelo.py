import numpy as np
import pytest
import torch

from qcnn_ecg.entrenamiento import ConfigEntrenamiento, entrenar_modelo, fijar_semillas
from qcnn_ecg.evaluacion import calcular_metricas, predecir_probabilidades, umbral_optimo_f1
from qcnn_ecg.modelos import CNN1D, ConfigCNN, contar_parametros


@pytest.mark.parametrize(
    ("config", "parametros"),
    [
        (ConfigCNN(), 11_265),
        (ConfigCNN(cabeza="flatten"), 82_945),
        (
            ConfigCNN(
                filtros=(32, 16), kernels=(15, 15), pools=(6, 6), cabeza="flatten", ocultas=10
            ),
            9_605,
        ),
    ],
)
def test_formas_y_parametros(config, parametros):
    modelo = CNN1D(config)
    assert modelo(torch.zeros(5, 1, 288)).shape == (5,)
    assert contar_parametros(modelo) == parametros


def test_config_invalida():
    with pytest.raises(ValueError):
        ConfigCNN(filtros=(16, 32), kernels=(7,), pools=(2, 2))


def test_umbral_optimo_caso_conocido():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    umbral, f1 = umbral_optimo_f1(y, p)
    assert umbral == pytest.approx(0.7) and f1 == pytest.approx(1.0)
    metricas = calcular_metricas(y, p, umbral)
    assert metricas["matriz_confusion"] == [[3, 0], [0, 2]]
    assert metricas["especificidad"] == 1.0 and metricas["exactitud"] == 1.0


def test_entrenamiento_aprende_patron_sintetico():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600).astype(np.int8)
    X = rng.normal(0, 1, (600, 288)).astype(np.float32)
    X[y == 1, 130:160] += 3.0  # la clase anormal tiene un "QRS" más ancho
    fijar_semillas(0)
    cfg = ConfigEntrenamiento(epocas_max=3, tamano_lote=64, hilos=2)
    modelo, historial = entrenar_modelo(
        CNN1D(), X[:400], y[:400], X[400:], y[400:], cfg, semilla=0, registrar=lambda _: None
    )
    p = predecir_probabilidades(modelo, X[400:])
    assert not np.isnan(p).any()
    assert len(historial) <= 3 and max(h["f1_val"] for h in historial) > 0.9
