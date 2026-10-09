"""CLI: entrena una variante de la CNN con una semilla y guarda sus resultados.

Uso:
    uv run python -m qcnn_ecg.entrenar --config configs/cnn/base.toml --semilla 0
    uv run python -m qcnn_ecg.entrenar --config configs/cnn/base.toml --semilla 0 --rapido
    uv run python -m qcnn_ecg.entrenar --config ... --semilla 0 --evaluar-prueba

El conjunto de prueba solo se evalúa con --evaluar-prueba (o desde
qcnn_ecg.experimentos para la configuración final), con el umbral congelado en
validación.
"""

import argparse
import json
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qcnn_ecg import config
from qcnn_ecg.conjunto import Conjunto, cargar_conjunto
from qcnn_ecg.entrenamiento import ConfigEntrenamiento, entrenar_modelo, fijar_semillas
from qcnn_ecg.evaluacion import (
    acierto_por_simbolo,
    calcular_metricas,
    medir_inferencia,
    predecir_probabilidades,
    umbral_optimo_f1,
)
from qcnn_ecg.graficos import graficar_curvas, graficar_matriz, graficar_roc_pr
from qcnn_ecg.modelos import CNN1D, ConfigCNN, contar_parametros

DIR_CORRIDAS = config.DIR_RESULTADOS / "cnn"
DIR_CORRIDAS_RAPIDAS = config.DIR_DATOS / "corridas_rapidas"


def leer_config(ruta: Path) -> tuple[str, ConfigCNN, ConfigEntrenamiento]:
    datos = tomllib.loads(Path(ruta).read_text(encoding="utf-8"))
    return (
        datos["nombre"],
        ConfigCNN.desde_dict(datos["modelo"]),
        ConfigEntrenamiento(**datos["entrenamiento"]),
    )


def _guardar_json(datos: dict, ruta: Path) -> None:
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _mejor_epoca(historial: list[dict]) -> int:
    suavizado = pd.DataFrame(historial)["f1_val_suavizado"]
    return int(suavizado.idxmax()) + 1 if suavizado.notna().any() else len(historial)


def ejecutar_corrida(
    ruta_config: Path,
    semilla: int,
    conjunto: Conjunto | None = None,
    rapido: bool = False,
    registrar=print,
) -> dict:
    """Entrena una variante, fija el umbral en validación y guarda todo en su carpeta."""
    nombre, cfg_modelo, cfg_ent = leer_config(ruta_config)
    conjunto = conjunto if conjunto is not None else cargar_conjunto()
    X_ent, y_ent = conjunto.subconjunto("entrenamiento")
    X_val, y_val = conjunto.subconjunto("validacion")
    if rapido:
        submuestra = np.random.default_rng(semilla).permutation(len(y_ent))[: len(y_ent) // 10]
        X_ent, y_ent = X_ent[submuestra], y_ent[submuestra]
        cfg_ent = ConfigEntrenamiento(**{**cfg_ent.a_dict(), "epocas_max": 1})

    directorio = (DIR_CORRIDAS_RAPIDAS if rapido else DIR_CORRIDAS) / f"{nombre}_s{semilla}"
    directorio.mkdir(parents=True, exist_ok=True)

    fijar_semillas(semilla)
    modelo = CNN1D(cfg_modelo)
    parametros = contar_parametros(modelo)
    registrar(f"[{nombre} | semilla {semilla}] {parametros:,} parámetros, {len(y_ent):,} latidos")

    inicio = time.perf_counter()
    modelo, historial = entrenar_modelo(
        modelo, X_ent, y_ent, X_val, y_val, cfg_ent, semilla, registrar
    )
    segundos = time.perf_counter() - inicio

    p_val = predecir_probabilidades(modelo, X_val)
    umbral, _ = umbral_optimo_f1(y_val, p_val)
    simbolos_val = conjunto.simbolo[conjunto.mascara("validacion")]
    resultado = {
        "nombre": nombre,
        "semilla": semilla,
        "rapido": rapido,
        "parametros": parametros,
        "epocas": len(historial),
        "mejor_epoca": _mejor_epoca(historial),
        "segundos_entrenamiento": segundos,
        "umbral": umbral,
        "validacion": calcular_metricas(y_val, p_val, umbral),
        "validacion_por_simbolo": acierto_por_simbolo(y_val, p_val, simbolos_val, umbral),
    }

    historial = pd.DataFrame(historial)
    historial.to_csv(directorio / "historial.csv", index=False)
    _guardar_json(
        {
            "nombre": nombre,
            "modelo": cfg_modelo.a_dict(),
            "entrenamiento": cfg_ent.a_dict(),
            "semilla": semilla,
            "conjunto": config.NOMBRE_CONJUNTO,
        },
        directorio / "config.json",
    )
    _guardar_json(resultado, directorio / "metricas.json")
    torch.save(
        {"estado": modelo.state_dict(), "config_modelo": cfg_modelo.a_dict(), "umbral": umbral},
        directorio / "modelo.pt",
    )
    graficar_curvas(historial, directorio / "curvas.png", f"{nombre} · semilla {semilla}")
    graficar_matriz(
        resultado["validacion"]["matriz_confusion"],
        directorio / "matriz_confusion_validacion.png",
        f"Validación · {nombre} · s{semilla}",
    )
    registrar(
        f"[{nombre} | semilla {semilla}] F1 val {resultado['validacion']['f1']:.4f} "
        f"(umbral {umbral:.3f}) en {segundos / 60:.1f} min"
    )
    resultado["directorio"] = str(directorio)
    return resultado


def cargar_modelo(ruta: Path) -> tuple[CNN1D, float]:
    """Carga un modelo guardado por ejecutar_corrida (o el modelo final)."""
    guardado = torch.load(ruta, map_location="cpu", weights_only=True)
    modelo = CNN1D(ConfigCNN.desde_dict(guardado["config_modelo"]))
    modelo.load_state_dict(guardado["estado"])
    modelo.eval()
    return modelo, float(guardado["umbral"])


def evaluar_prueba(directorio: Path, conjunto: Conjunto | None = None, registrar=print) -> dict:
    """Evalúa UNA vez en prueba (DS2 sin 202) con el umbral fijado en validación.

    Reporta además DS2 con el 202 incluido y el tiempo de inferencia.
    """
    directorio = Path(directorio)
    conjunto = conjunto if conjunto is not None else cargar_conjunto()
    modelo, umbral = cargar_modelo(directorio / "modelo.pt")
    torch.set_num_threads(4)

    m_prueba = conjunto.mascara("prueba")
    m_con_202 = m_prueba | conjunto.mascara("prueba_excluida")
    p = predecir_probabilidades(modelo, conjunto.X[m_con_202])
    y = conjunto.y[m_con_202]
    es_prueba = m_prueba[m_con_202]

    resultado = {
        "umbral": umbral,
        "prueba": calcular_metricas(y[es_prueba], p[es_prueba], umbral),
        "prueba_por_simbolo": acierto_por_simbolo(
            y[es_prueba], p[es_prueba], conjunto.simbolo[m_prueba], umbral
        ),
        "prueba_con_202": calcular_metricas(y, p, umbral),
        "inferencia": medir_inferencia(modelo, conjunto.X[m_prueba]),
    }
    _guardar_json(resultado, directorio / "metricas_prueba.json")
    graficar_matriz(
        resultado["prueba"]["matriz_confusion"],
        directorio / "matriz_confusion_prueba.png",
        f"Prueba (DS2 sin 202) · {directorio.name}",
    )
    graficar_roc_pr(y[es_prueba], p[es_prueba], directorio / "roc_pr_prueba.png", directorio.name)
    registrar(
        f"[{directorio.name}] PRUEBA: F1 {resultado['prueba']['f1']:.4f}, "
        f"exactitud {resultado['prueba']['exactitud']:.4f}, "
        f"sensibilidad {resultado['prueba']['sensibilidad']:.4f}"
    )
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=config.DIR_CONFIGS / "cnn" / "base.toml")
    parser.add_argument("--semilla", type=int, default=0)
    parser.add_argument(
        "--rapido", action="store_true", help="1 época con el 10 %% del entrenamiento"
    )
    parser.add_argument("--evaluar-prueba", action="store_true")
    args = parser.parse_args()

    resultado = ejecutar_corrida(args.config, args.semilla, rapido=args.rapido)
    if args.evaluar_prueba and not args.rapido:
        evaluar_prueba(Path(resultado["directorio"]))


if __name__ == "__main__":
    main()
