import numpy as np
import pytest

from app.csv_ecg import leer_senal
from app.errores import ErrorAPI


def construir_csv(n=3600, fs=360.0, amplitud=1.0, encabezado="tiempo_s,mlii_mv", fin="\n"):
    """CSV sintético válido: seno de 1 Hz. n=3600 a 360 Hz son justo 10 s."""
    t = np.arange(n) / fs
    y = amplitud * np.sin(2 * np.pi * t)
    filas = [f"{a:.6f},{b:.3f}" for a, b in zip(t, y)]
    return encabezado + fin + fin.join(filas) + fin


def codigo_de(texto):
    with pytest.raises(ErrorAPI) as e:
        leer_senal(texto)
    return e.value.codigo


def test_valido():
    s = leer_senal(construir_csv())
    assert s.muestras == 3600
    assert abs(s.fs_hz - 360) < 0.01
    assert abs(s.duracion_s - 10) < 0.01


def test_valido_con_saltos_de_linea_de_windows():
    leer_senal(construir_csv(fin="\r\n"))


def test_columnas_invalidas():
    assert codigo_de(construir_csv(encabezado="tiempo,mv")) == "COLUMNAS_INVALIDAS"


def test_valor_no_numerico_indica_la_fila():
    lineas = construir_csv().split("\n")
    lineas[4] = "0.011111,abc"  # línea 5 del archivo
    with pytest.raises(ErrorAPI) as e:
        leer_senal("\n".join(lineas))
    assert e.value.codigo == "VALORES_INVALIDOS"
    assert e.value.detalle["fila"] == 5


def test_celda_vacia():
    lineas = construir_csv().split("\n")
    lineas[10] = "0.027778,"
    assert codigo_de("\n".join(lineas)) == "VALORES_INVALIDOS"


def test_nan_literal():
    lineas = construir_csv().split("\n")
    lineas[10] = "0.027778,NaN"
    assert codigo_de("\n".join(lineas)) == "VALORES_INVALIDOS"


def test_fila_con_tres_columnas():
    lineas = construir_csv().split("\n")
    lineas[10] = "0.027778,0.5,9"
    assert codigo_de("\n".join(lineas)) == "VALORES_INVALIDOS"


def test_solo_encabezado():
    assert codigo_de("tiempo_s,mlii_mv\n") == "VALORES_INVALIDOS"


def test_tiempo_no_creciente():
    lineas = construir_csv().split("\n")
    lineas[20] = lineas[19]  # tiempo repetido
    assert codigo_de("\n".join(lineas)) == "TIEMPO_NO_CRECIENTE"


def test_muestreo_irregular():
    lineas = construir_csv().split("\n")
    del lineas[100]  # falta una muestra: un paso se duplica
    assert codigo_de("\n".join(lineas)) == "MUESTREO_IRREGULAR"


def test_frecuencia_baja_y_alta():
    assert codigo_de(construir_csv(n=1000, fs=50)) == "FRECUENCIA_FUERA_DE_RANGO"
    assert codigo_de(construir_csv(n=20000, fs=2000)) == "FRECUENCIA_FUERA_DE_RANGO"


def test_duracion_corta_y_larga():
    assert codigo_de(construir_csv(n=2232)) == "DURACION_INSUFICIENTE"  # 6.2 s
    assert codigo_de(construir_csv(n=360 * 301)) == "DURACION_EXCESIVA"


def test_limites_exactos_de_duracion():
    leer_senal(construir_csv(n=3600))  # 10 s exactos
    leer_senal(construir_csv(n=108000))  # 300 s exactos (5 min)


def test_amplitud_fuera_de_rango():
    assert codigo_de(construir_csv(amplitud=50)) == "AMPLITUD_FUERA_DE_RANGO"


def test_otras_frecuencias_validas():
    for fs in (100, 250, 500, 1000):
        s = leer_senal(construir_csv(n=int(fs * 12), fs=fs))
        assert abs(s.fs_hz - fs) < 0.5