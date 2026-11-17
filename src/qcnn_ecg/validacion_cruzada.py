"""Validación cruzada por paciente en DS1; el modelo final es el ensamble de sus 5 modelos.

Con un único pliegue de validación (5 pacientes) la selección de variante y de
umbral quedó dominada por un solo paciente (207) y el umbral no se trasladó a la
prueba. Aquí cada variante se entrena en 5 pliegues de pacientes de DS1:

- las predicciones fuera de pliegue (OOF) de los 22 pacientes de DS1 se juntan
  para medir la variante y fijar el umbral que maximiza el F1;
- el modelo final es el **ensamble de los 5 modelos de pliegue** (promedio de probabilidades)
  con ese mismo umbral. Así el umbral se fijó con los mismos modelos a los que se aplica
  (en la v1 se calculaba con los de la CV pero se aplicaba a otro modelo entrenado con todo DS1).
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qcnn_ecg.conjunto import Conjunto
from qcnn_ecg.entrenamiento import entrenar_modelo, fijar_semillas
from qcnn_ecg.entrenar import DIR_CORRIDAS, construir_modelo, leer_config
from qcnn_ecg.evaluacion import (
    acierto_por_simbolo,
    calcular_metricas,
    predecir_probabilidades,
    umbral_optimo_f1,
)
from qcnn_ecg.graficos import graficar_curvas, graficar_matriz
from qcnn_ecg.modelos import contar_parametros
from qcnn_ecg.particion import pliegues_ds1

DIR_CV = DIR_CORRIDAS / "cv"


def mascara_ds1(conjunto: Conjunto) -> np.ndarray:
    return conjunto.mascara("entrenamiento") | conjunto.mascara("validacion")


def _guardar_json(datos: dict, ruta: Path) -> None:
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ejecutar_cv(
    ruta_config: Path, semilla: int, conjunto: Conjunto, registrar=print, dir_cv: Path = DIR_CV
) -> dict:
    """Validación cruzada de 5 pliegues por paciente en DS1 para una variante."""
    nombre, cfg_modelo, cfg_ent = leer_config(ruta_config)
    directorio = dir_cv / f"{nombre}_s{semilla}"
    directorio.mkdir(parents=True, exist_ok=True)

    ds1 = mascara_ds1(conjunto)
    X, y = conjunto.X[ds1], conjunto.y[ds1]
    rr = conjunto.rr[ds1]
    registros, simbolos = conjunto.registro[ds1], conjunto.simbolo[ds1]
    pliegues = pliegues_ds1(registros, y)

    p_oof = np.full(len(y), np.nan)
    por_pliegue, estados, inicio = [], [], time.perf_counter()
    for k, regs_val in enumerate(pliegues):
        val = np.isin(registros, regs_val)
        registrar(f"[{nombre} | s{semilla} | pliegue {k}] validación: {', '.join(regs_val)}")
        fijar_semillas(semilla)
        modelo, historial = entrenar_modelo(
            construir_modelo(cfg_modelo),
            X[~val],
            y[~val],
            X[val],
            y[val],
            cfg_ent,
            semilla,
            registrar,
            rr[~val],
            rr[val],
        )
        estados.append(modelo.state_dict())
        p_oof[val] = predecir_probabilidades(modelo, X[val], rr=rr[val])
        historial = pd.DataFrame(historial)
        historial.to_csv(directorio / f"historial_pliegue{k}.csv", index=False)
        graficar_curvas(historial, directorio / f"curvas_pliegue{k}.png", f"{nombre} · pliegue {k}")
        suavizado = historial["f1_val_suavizado"]
        mejor_epoca = int(suavizado.idxmax()) + 1 if suavizado.notna().any() else len(historial)
        umbral_k, f1_k = umbral_optimo_f1(y[val], p_oof[val])
        por_pliegue.append(
            {
                "pliegue": k,
                "registros": list(regs_val),
                "latidos": int(val.sum()),
                "epocas": len(historial),
                "mejor_epoca": mejor_epoca,
                "f1_umbral_propio": f1_k,
                "umbral_propio": umbral_k,
                "auc_pr": calcular_metricas(y[val], p_oof[val])["auc_pr"],
            }
        )

    umbral, _ = umbral_optimo_f1(y, p_oof)
    resultado = {
        "nombre": nombre,
        "semilla": semilla,
        "parametros": contar_parametros(construir_modelo(cfg_modelo)),
        "minutos": (time.perf_counter() - inicio) / 60,
        "umbral_oof": umbral,
        "mejor_epoca_mediana": float(np.median([p["mejor_epoca"] for p in por_pliegue])),
        "oof": calcular_metricas(y, p_oof, umbral),
        "oof_por_simbolo": acierto_por_simbolo(y, p_oof, simbolos, umbral),
        "pliegues": por_pliegue,
    }
    np.savez_compressed(directorio / "oof.npz", p=p_oof, y=y, registro=registros, simbolo=simbolos)
    _guardar_json(
        {"nombre": nombre, "modelo": cfg_modelo.a_dict(), "entrenamiento": cfg_ent.a_dict()},
        directorio / "config.json",
    )
    torch.save(
        {"estados": estados, "config_modelo": cfg_modelo.a_dict(), "umbral": umbral},
        directorio / "modelo.pt",
    )
    _guardar_json(resultado, directorio / "metricas_cv.json")
    graficar_matriz(
        resultado["oof"]["matriz_confusion"],
        directorio / "matriz_confusion_oof.png",
        f"DS1 fuera de pliegue · {nombre}",
    )
    registrar(
        f"[{nombre} | s{semilla}] CV: F1 OOF {resultado['oof']['f1']:.4f}"
        f" (umbral {umbral:.3f}), AUC-PR {resultado['oof']['auc_pr']:.4f},"
        f" mejor época mediana {resultado['mejor_epoca_mediana']:.0f}, {resultado['minutos']:.1f} min"
    )
    resultado["directorio"] = str(directorio)
    return resultado
