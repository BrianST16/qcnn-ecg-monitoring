"""Constantes del proyecto: rutas, preprocesamiento, etiquetas y partición.

Todas las piezas del proyecto (CNN, QCNN y prototipo) deben leer estos valores
desde aquí para garantizar condiciones experimentales idénticas.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

RAIZ = Path(__file__).resolve().parents[2]
DIR_DATOS = RAIZ / "data"
DIR_MITDB = DIR_DATOS / "raw" / "mitdb"
DIR_PROCESADOS = DIR_DATOS / "processed"
DIR_MANIFIESTOS = DIR_DATOS / "manifiestos"
DIR_CONFIGS = RAIZ / "configs"
DIR_RESULTADOS = RAIZ / "results"
DIR_MODELOS = RAIZ / "modelos_entrenados"

NOMBRE_CONJUNTO = "mitdb_binario_v1"

# ---------------------------------------------------------------------------
# Señal y preprocesamiento
# ---------------------------------------------------------------------------

FS = 360  # Hz, frecuencia de muestreo de MIT-BIH
DERIVACION = "MLII"  # se busca por nombre: en el registro 114 está en el canal 1
BANDA_HZ = (0.5, 40.0)  # pasa-banda: quita deriva de línea base y ruido de red
ORDEN_FILTRO = 4
MEDIA_VENTANA = 144  # 0.4 s a 360 Hz antes y después del pico R
LONGITUD_VENTANA = 2 * MEDIA_VENTANA  # 288 muestras = 0.8 s

# ---------------------------------------------------------------------------
# Etiquetas (binario: 0 = normal, 1 = anormal)
# ---------------------------------------------------------------------------

SIMBOLOS_NORMAL = ("N",)
SIMBOLOS_ANORMAL = ("L", "R", "A", "a", "J", "S", "V", "E", "F", "j", "e")
NOMBRES_CLASES = ("NORMAL", "ANORMAL")

# ---------------------------------------------------------------------------
# Partición inter-paciente (de Chazal et al., 2004)
# ---------------------------------------------------------------------------

# fmt: off
DS1 = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
    "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
)
DS2 = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
    "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
)
# fmt: on
MARCAPASOS = ("102", "104", "107", "217")  # excluidos de todo el estudio
EXCLUIDOS_PRUEBA = ("202",)  # misma persona que el 201 (DS1); se reporta aparte

SEMILLA_PARTICION = 42

PARTICIONES = ("entrenamiento", "validacion", "prueba", "prueba_excluida")
