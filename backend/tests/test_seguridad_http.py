import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.limitador import LimitadorPorIP
from app.main import app

CLAVE = os.environ["API_KEY"]
CABECERA = {"X-API-Key": CLAVE}
EJEMPLO = Path(__file__).resolve().parents[2] / "Archivos" / "ejemplo_ecg.csv"
hay_ejemplo = pytest.mark.skipif(not EJEMPLO.exists(), reason="falta Archivos/ejemplo_ecg.csv")


@pytest.fixture()
def cliente():
    # base_url con "localhost": TestClient usa "testserver", que la capa de hosts rechazaría.
    # El bloque `with` ejecuta el lifespan, así cada prueba empieza con cupo y limitador nuevos.
    with TestClient(app, base_url="http://localhost") as c:
        yield c


def subir_ejemplo(cliente, cabecera=CABECERA, **datos):
    contenido = EJEMPLO.read_bytes()
    return cliente.post(
        "/api/v1/analisis",
        headers=cabecera,
        files={"archivo": ("ejemplo_ecg.csv", contenido, "text/csv")},
        data=datos,
    )


# ---------- API key ----------
def test_salud_es_publica(cliente):
    assert cliente.get("/api/v1/salud").status_code == 200


@pytest.mark.parametrize("cabecera", [{}, {"X-API-Key": "clave-incorrecta"}])
def test_modelos_exige_clave(cliente, cabecera):
    r = cliente.get("/api/v1/modelos", headers=cabecera)
    assert r.status_code == 401
    assert r.json()["error"]["codigo"] == "NO_AUTORIZADO"


def test_modelos_con_clave_correcta(cliente):
    assert cliente.get("/api/v1/modelos", headers=CABECERA).status_code == 200


def test_analisis_exige_clave(cliente):
    r = cliente.post(
        "/api/v1/analisis", files={"archivo": ("a.csv", b"tiempo_s,mlii_mv\n", "text/csv")}
    )
    assert r.status_code == 401
    assert r.json()["error"]["codigo"] == "NO_AUTORIZADO"


def test_los_errores_no_revelan_la_clave(cliente):
    r = cliente.get("/api/v1/modelos", headers={"X-API-Key": "clave-incorrecta"})
    assert CLAVE not in r.text


# ---------- Hosts y CORS ----------
def test_host_no_permitido_se_rechaza(cliente):
    r = cliente.get("/api/v1/salud", headers={"Host": "sitio-malicioso.example"})
    assert r.status_code == 400


def test_cors_permite_solo_el_origen_configurado(cliente):
    permitido = cliente.get("/api/v1/salud", headers={"Origin": "http://localhost:5173"})
    ajeno = cliente.get("/api/v1/salud", headers={"Origin": "http://sitio-ajeno.example"})
    assert permitido.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in ajeno.headers


# ---------- Tamaño ----------
def test_cuerpo_de_6_mb_se_corta_con_413(cliente):
    grande = b"a" * (6 * 1024 * 1024)
    r = cliente.post(
        "/api/v1/analisis",
        headers=CABECERA,
        files={"archivo": ("grande.csv", grande, "text/csv")},
    )
    assert r.status_code == 413
    assert r.json()["error"]["codigo"] == "ARCHIVO_DEMASIADO_GRANDE"


# ---------- Límite por IP ----------
def test_limite_de_peticiones_por_ip(cliente):
    cliente.app.state.limitador = LimitadorPorIP(3)
    codigos = [
        cliente.get("/api/v1/modelos", headers={"X-API-Key": "mala"}).status_code
        for _ in range(4)
    ]
    assert codigos == [401, 401, 401, 429]
    r = cliente.get("/api/v1/modelos", headers={"X-API-Key": "mala"})
    assert r.json()["error"]["codigo"] == "DEMASIADAS_PETICIONES"
    assert r.json()["error"]["detalle"]["reintentar_en_s"] >= 1
    assert cliente.get("/api/v1/salud").status_code == 200  # /salud no tiene límite


# ---------- Entradas inválidas y errores genéricos ----------
def test_modelos_desconocidos_se_rechazan(cliente):
    r = cliente.post(
        "/api/v1/analisis",
        headers=CABECERA,
        files={"archivo": ("a.csv", b"tiempo_s,mlii_mv\n", "text/csv")},
        data={"modelos": "cnn,gpt"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["codigo"] == "SOLICITUD_INVALIDA"


def test_ruta_inexistente_devuelve_el_formato_de_error(cliente):
    r = cliente.get("/api/v1/no-existe")
    assert r.status_code == 404
    assert "codigo" in r.json()["error"]


@hay_ejemplo
def test_analisis_valido_trae_la_advertencia(cliente):
    r = subir_ejemplo(cliente)
    assert r.status_code == 200
    cuerpo = r.json()
    assert "No constituye un diagnóstico médico" in cuerpo["advertencia"]
    assert cuerpo["resultados"]["cnn"]["estado"] == "ok"
    assert cuerpo["resultados"]["qcnn"]["estado"] == "no_disponible"


@hay_ejemplo
def test_un_modelo_que_falla_no_filtra_detalles(cliente):
    class ModeloRoto:
        clave = "cnn"
        id = "roto"
        nombre = "Roto"
        parametros = 1
        umbral = 0.5

        def probabilidades(self, ventanas):
            raise RuntimeError("dato-secreto-del-paciente")

    cliente.app.state.modelos["cnn"] = ModeloRoto()
    r = subir_ejemplo(cliente)
    assert r.status_code == 200
    assert r.json()["resultados"]["cnn"]["estado"] == "error"
    assert "dato-secreto-del-paciente" not in r.text