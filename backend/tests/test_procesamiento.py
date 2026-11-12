import numpy as np
import pytest

from app.csv_ecg import SenalLeida
from app.errores import ErrorAPI
from app.procesamiento import procesar_senal, remuestrear_a_modelo


def construir_senal(n=3600, fs=360.0, t0=0.0):
    """Señal sintética (seno de 1 Hz) con la forma que devuelve leer_senal."""
    tiempo = t0 + np.arange(n) / fs
    mv = np.sin(2 * np.pi * np.arange(n) / fs)
    return SenalLeida(tiempo_s=tiempo, mv=mv, fs_hz=fs, duracion_s=round(n / fs, 2))


def detector_fijo(*picos):
    """Detector falso: devuelve siempre los mismos picos (así no hace falta wfdb)."""
    return lambda x: np.array(picos, dtype=np.int64)


def test_no_remuestrea_si_ya_esta_a_360_hz():
    x = np.zeros(100)
    assert remuestrear_a_modelo(x, 360.0)[1] is False
    assert remuestrear_a_modelo(x, 360.0000089)[1] is False  # la fs deducida casi nunca es exacta


@pytest.mark.parametrize("fs", [100, 250, 500, 1000])
def test_remuestreo_deja_la_duracion_igual(fs):
    n = fs * 12
    y, remuestreada = remuestrear_a_modelo(np.sin(np.arange(n)), fs)
    assert remuestreada is True
    assert len(y) == 4320  # 12 s a 360 Hz


def test_descarta_picos_en_los_bordes():
    # Señal de 10 s (3600 muestras). Una ventana necesita 144 muestras a cada lado.
    r = procesar_senal(construir_senal(), detector_fijo(50, 400, 800, 1200, 1600, 3500))
    assert r.latidos_detectados == 6
    assert r.latidos_evaluados == 4  # se caen el 50 (borde izquierdo) y el 3500 (borde derecho)
    assert r.latidos_descartados_borde == 2
    assert r.ventanas.shape == (4, 288)
    assert np.allclose(r.tiempos_s, [400 / 360, 800 / 360, 1200 / 360, 1600 / 360])


def test_acepta_picos_justo_en_el_limite():
    r = procesar_senal(construir_senal(), detector_fijo(144, 500, 1000, 1500, 3456))
    assert r.latidos_evaluados == 5  # 144 y 3600-144 todavía caben


def test_rr_y_frecuencia_cardiaca():
    r = procesar_senal(construir_senal(), detector_fijo(400, 760, 1120, 1480, 1840))  # 1 s entre latidos
    assert np.isnan(r.rr_previo_s[0])  # el primer latido no tiene anterior
    assert np.allclose(r.rr_previo_s[1:], 1.0)
    assert abs(r.fc_media_lpm - 60.0) < 1e-9


def test_el_rr_usa_tambien_el_latido_descartado_por_el_borde():
    # El pico 50 se descarta, pero sirve como "anterior" del pico 410: RR = 360/360 = 1 s.
    r = procesar_senal(construir_senal(), detector_fijo(50, 410, 770, 1130, 1490))
    assert r.latidos_evaluados == 4
    assert abs(r.rr_previo_s[0] - 1.0) < 1e-9


def test_tiempo_en_la_escala_del_csv():
    # Si el CSV empieza en t = 5 s, los tiempos de los latidos también.
    r = procesar_senal(construir_senal(t0=5.0), detector_fijo(360, 720, 1080, 1440, 1800))
    assert np.allclose(r.tiempos_s, 5.0 + np.array([360, 720, 1080, 1440, 1800]) / 360)


def test_pocos_latidos():
    with pytest.raises(ErrorAPI) as e:
        procesar_senal(construir_senal(), detector_fijo(400, 800, 1200, 1600))
    assert e.value.codigo == "SIN_LATIDOS_SUFICIENTES"
    assert e.value.detalle["latidos_detectados"] == 4


def test_sin_ningun_pico():
    with pytest.raises(ErrorAPI) as e:
        procesar_senal(construir_senal(), detector_fijo())
    assert e.value.codigo == "SIN_LATIDOS_SUFICIENTES"


def test_remuestrea_y_marca_la_senal():
    r = procesar_senal(construir_senal(n=3000, fs=250.0), detector_fijo(400, 800, 1200, 1600, 2000))
    assert r.remuestreada is True
    assert abs(r.fs_original_hz - 250.0) < 1e-9