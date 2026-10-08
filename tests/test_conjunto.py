import numpy as np

from qcnn_ecg import config
from qcnn_ecg.conjunto import cargar_conjunto, guardar_arrays, procesar_registros

from .conftest import requiere_conjunto, requiere_mitdb


@requiere_mitdb
def test_procesamiento_reproducible(tmp_path):
    a = procesar_registros(["100", "114"])
    b = procesar_registros(["100", "114"])
    assert guardar_arrays(a, tmp_path / "a") == guardar_arrays(b, tmp_path / "b")
    assert a["X"].shape[1] == config.LONGITUD_VENTANA and a["X"].dtype == np.float32
    assert set(np.unique(a["y"])) <= {0, 1}


@requiere_conjunto
def test_conjunto_sin_fuga_de_pacientes():
    conjunto = cargar_conjunto()
    por_particion = {
        p: set(conjunto.registro[conjunto.mascara(p)].tolist()) for p in config.PARTICIONES
    }
    vistos = set()
    for registros in por_particion.values():
        assert not registros & vistos
        vistos |= registros
    assert "202" not in por_particion["prueba"]
    assert not vistos & set(config.MARCAPASOS)
    assert por_particion["validacion"] == {"106", "108", "116", "205", "207"}
    assert not np.isnan(conjunto.X).any()
