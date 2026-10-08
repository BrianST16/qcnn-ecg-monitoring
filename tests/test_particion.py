import numpy as np

from qcnn_ecg import config
from qcnn_ecg.datos import leer_anotaciones, leer_registro
from qcnn_ecg.particion import asignar_particiones, elegir_validacion
from qcnn_ecg.preprocesamiento import etiquetar_simbolo

from .conftest import requiere_mitdb


def test_ds1_ds2_disjuntos_y_sin_marcapasos():
    assert not set(config.DS1) & set(config.DS2)
    assert not set(config.MARCAPASOS) & (set(config.DS1) | set(config.DS2))
    assert len(config.DS1) == len(config.DS2) == 22


def test_asignacion_excluye_202_de_prueba():
    asignacion = asignar_particiones(("106", "108"))
    assert asignacion["202"] == "prueba_excluida"
    assert asignacion["201"] == "entrenamiento"
    assert asignacion["106"] == "validacion"
    assert all(asignacion[r] == "prueba" for r in config.DS2 if r != "202")


@requiere_mitdb
def test_validacion_determinista_y_esperada():
    registros, y, simbolos = [], [], []
    for r in config.DS1:
        muestras, sims = leer_anotaciones(r)
        for m, s in zip(muestras, sims, strict=True):
            etiqueta = etiquetar_simbolo(s)
            if etiqueta is not None and config.MEDIA_VENTANA <= m <= 650000 - config.MEDIA_VENTANA:
                registros.append(r)
                y.append(etiqueta)
                simbolos.append(s)
    arrays = np.array(registros), np.array(y), np.array(simbolos)
    validacion = elegir_validacion(*arrays)
    assert validacion == ("106", "108", "116", "205", "207")
    assert elegir_validacion(*arrays) == validacion


@requiere_mitdb
def test_mlii_por_nombre_en_registro_114():
    registro = leer_registro("114")
    assert registro.derivacion == "MLII" and registro.indice_canal == 1
    assert all(leer_registro(r).derivacion == "MLII" for r in ("100", "201"))
