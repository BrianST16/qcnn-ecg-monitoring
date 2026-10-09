import numpy as np
import pytest

from qcnn_ecg.config import FS, LONGITUD_VENTANA
from qcnn_ecg.preprocesamiento import (
    etiquetar_simbolo,
    extraer_ventanas,
    filtrar_pasabanda,
    normalizar_registro,
)


def test_filtro_conserva_longitud_y_quita_deriva():
    t = np.arange(10 * FS) / FS
    deriva = 2.0 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz, debajo de la banda
    util = np.sin(2 * np.pi * 10 * t)  # 10 Hz, dentro de la banda
    assert filtrar_pasabanda(deriva + util).shape == t.shape
    centro = slice(2 * FS, 8 * FS)  # evita efectos de borde
    assert np.abs(filtrar_pasabanda(deriva)[centro]).max() < 0.01 * 2.0  # deriva atenuada >99 %
    amplitud_util = np.abs(filtrar_pasabanda(util)[centro]).max()
    assert 0.9 < amplitud_util < 1.1  # la banda útil se conserva


def test_normalizacion_media_cero_desviacion_uno():
    x = np.random.default_rng(0).normal(3.0, 2.0, 5000)
    z = normalizar_registro(x)
    assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-9


def test_normalizacion_senal_constante():
    assert np.all(normalizar_registro(np.full(10, 4.0)) == 0)


@pytest.mark.parametrize(
    ("simbolo", "esperado"),
    [
        ("N", 0),
        ("V", 1),
        ("L", 1),
        ("R", 1),
        ("A", 1),
        ("e", 1),
        ("j", 1),
        ("Q", None),
        ("/", None),
        ("f", None),
        ("+", None),
        ("~", None),
    ],
)
def test_mapeo_etiquetas(simbolo, esperado):
    assert etiquetar_simbolo(simbolo) == esperado


def test_ventanas_centradas_y_bordes_descartados():
    senal = np.arange(1000, dtype=float)
    ventanas, mascara = extraer_ventanas(senal, np.array([10, 144, 500, 856, 990]))
    assert mascara.tolist() == [False, True, True, True, False]
    assert ventanas.shape == (3, LONGITUD_VENTANA)
    assert ventanas[1, 144] == 500  # el pico R queda en la posición 144
