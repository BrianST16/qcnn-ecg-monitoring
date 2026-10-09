"""Partición inter-paciente: DS1 (entrenamiento + validación) y DS2 (prueba).

La validación se separa por paciente dentro de DS1 de forma determinista: se
recorren los pliegues de StratifiedGroupKFold (semilla fija) y se toma el primero
que cumpla restricciones de tamaño, proporción de clases y cobertura de símbolos.
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from qcnn_ecg.config import DS1, DS2, EXCLUIDOS_PRUEBA, SEMILLA_PARTICION

FRACCION_VALIDACION = (0.15, 0.25)  # fracción de latidos de DS1
PROPORCION_ANORMAL_VALIDACION = (0.18, 0.33)
SIMBOLOS_REQUERIDOS_VALIDACION = ("V", "A")  # además de al menos uno de L o R
SIMBOLOS_PROTEGIDOS_ENTRENAMIENTO = ("L", "R", "V", "A", "F")  # >= 50 % queda en entrenamiento
INTENTOS_SEMILLA = 100


def _cumple_restricciones(
    entrenamiento: np.ndarray, validacion: np.ndarray, y: np.ndarray, simbolos: np.ndarray
) -> bool:
    fraccion = len(validacion) / len(y)
    proporcion = y[validacion].mean()
    sim_val = simbolos[validacion]
    sim_ent = simbolos[entrenamiento]
    return (
        FRACCION_VALIDACION[0] <= fraccion <= FRACCION_VALIDACION[1]
        and PROPORCION_ANORMAL_VALIDACION[0] <= proporcion <= PROPORCION_ANORMAL_VALIDACION[1]
        and all((sim_val == s).any() for s in SIMBOLOS_REQUERIDOS_VALIDACION)
        and ((sim_val == "L").any() or (sim_val == "R").any())
        and all(
            (sim_ent == s).sum() >= 0.5 * (simbolos == s).sum()
            for s in SIMBOLOS_PROTEGIDOS_ENTRENAMIENTO
        )
    )


def elegir_validacion(
    registros: np.ndarray, y: np.ndarray, simbolos: np.ndarray, semilla: int = SEMILLA_PARTICION
) -> tuple[str, ...]:
    """Elige los registros de validación dentro de DS1 (arrays alineados por latido)."""
    for intento in range(INTENTOS_SEMILLA):
        divisor = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=semilla + intento)
        for entrenamiento, validacion in divisor.split(simbolos, y, registros):
            if _cumple_restricciones(entrenamiento, validacion, y, simbolos):
                return tuple(sorted(set(registros[validacion].tolist())))
    raise RuntimeError("Ningún pliegue de validación cumple las restricciones")


def asignar_particiones(validacion: tuple[str, ...]) -> dict[str, str]:
    """Mapa registro -> partición para todos los registros del estudio."""
    asignacion = {r: ("validacion" if r in validacion else "entrenamiento") for r in DS1}
    for r in DS2:
        asignacion[r] = "prueba_excluida" if r in EXCLUIDOS_PRUEBA else "prueba"
    return asignacion


def pliegues_ds1(
    registros: np.ndarray, y: np.ndarray, semilla: int = SEMILLA_PARTICION, n_pliegues: int = 5
) -> list[tuple[str, ...]]:
    """Pliegues de validación cruzada por paciente sobre DS1 (registros de cada pliegue).

    Cada registro aparece en exactamente un pliegue; los pliegues se estratifican por
    la proporción de latidos anormales.
    """
    divisor = StratifiedGroupKFold(n_splits=n_pliegues, shuffle=True, random_state=semilla)
    return [
        tuple(sorted(set(registros[validacion].tolist())))
        for _, validacion in divisor.split(registros, y, registros)
    ]
