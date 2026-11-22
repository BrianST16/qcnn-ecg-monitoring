"""CLI: umbral del ensamble sin contaminación con CV anidada (<resultados>/anidado/).

Problema: el umbral OOF de `validacion_cruzada` sale de modelos INDIVIDUALES, pero el modelo final
es un ENSAMBLE que promedia probabilidades (las comprime hacia el centro). Promediar los 5 modelos
sobre DS1 no sirve: 4 de los 5 ya entrenaron con cada latido.

Solución: para cada pliegue externo k (pacientes que NINGÚN modelo de ese paso ve), se entrenan 4
modelos con los otros 4 pliegues (cada uno con uno de ellos como validación para la parada
temprana) y su ensamble predice el pliegue k. Los 5 pliegues juntos dan predicciones de ensamble
limpias para los 22 pacientes de DS1; de ahí sale el umbral. Aproximación: esos ensambles tienen
4 miembros entrenados con el 60 % de DS1; el modelo final tiene 5 miembros (los de la CV, con el
80 %) y conserva sus pesos: solo cambia el umbral.

Uso: uv run python -m qcnn_ecg.umbral_anidado --semilla 1
     uv run python -m qcnn_ecg.umbral_anidado --config configs/cnn/base_rr.toml
         --resultados results/cnn_v2 --semilla 0
Se puede interrumpir y repetir: reutiliza los modelos ya entrenados.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qcnn_ecg import config
from qcnn_ecg.conjunto import cargar_conjunto
from qcnn_ecg.entrenamiento import entrenar_modelo, fijar_semillas
from qcnn_ecg.entrenar import construir_modelo, evaluar_prueba, leer_config
from qcnn_ecg.evaluacion import calcular_metricas, predecir_probabilidades, umbral_optimo_f1
from qcnn_ecg.modelos import Ensamble
from qcnn_ecg.particion import pliegues_ds1
from qcnn_ecg.validacion_cruzada import mascara_ds1

DIR_QCNN = config.DIR_RESULTADOS / "qcnn"


def ejecutar(
    ruta_config: Path, semilla: int, registrar=print, base: Path = DIR_QCNN
) -> dict:
    nombre, cfg_modelo, cfg_ent = leer_config(ruta_config)
    cv = base / "cv" / f"{nombre}_s{semilla}"
    salida = base / "anidado" / f"{nombre}_s{semilla}"
    (salida / "modelos").mkdir(parents=True, exist_ok=True)

    conjunto = cargar_conjunto()
    ds1 = mascara_ds1(conjunto)
    X, y, rr = conjunto.X[ds1], conjunto.y[ds1], conjunto.rr[ds1]
    registros = conjunto.registro[ds1]
    pliegues = pliegues_ds1(registros, y)
    en_pliegue = [np.isin(registros, p) for p in pliegues]

    p_ens = np.full(len(y), np.nan)
    inicio = time.perf_counter()
    for k in range(len(pliegues)):
        miembros = []
        for j in (i for i in range(len(pliegues)) if i != k):
            ruta = salida / "modelos" / f"externo{k}_interno{j}.pt"
            modelo = construir_modelo(cfg_modelo)
            if ruta.exists():
                modelo.load_state_dict(torch.load(ruta, weights_only=True))
            else:
                registrar(f"[s{semilla}] externo {k} (fuera: {', '.join(pliegues[k])}), val {j}")
                entren = ~en_pliegue[k] & ~en_pliegue[j]
                fijar_semillas(semilla)
                modelo, _ = entrenar_modelo(
                    modelo, X[entren], y[entren], X[en_pliegue[j]], y[en_pliegue[j]],
                    cfg_ent, semilla, registrar, rr[entren], rr[en_pliegue[j]],
                )  # fmt: skip
                torch.save(modelo.state_dict(), ruta)
            miembros.append(modelo.eval())
            minutos = (time.perf_counter() - inicio) / 60
            registrar(f"  modelo {len(miembros)}/4 listo ({minutos:.1f} min)")
        p_ens[en_pliegue[k]] = predecir_probabilidades(
            Ensamble(miembros), X[en_pliegue[k]], rr=rr[en_pliegue[k]]
        )

    umbral, _ = umbral_optimo_f1(y, p_ens)
    guardado = torch.load(cv / "modelo.pt", weights_only=True)
    umbral_individual = guardado["umbral"]
    guardado["umbral"] = umbral
    torch.save(guardado, salida / "modelo.pt")
    resultado = {
        "semilla": semilla,
        "umbral_ensamble_anidado": umbral,
        "umbral_individual_cv": umbral_individual,
        "oof_ensamble": calcular_metricas(y, p_ens, umbral),
        "minutos": (time.perf_counter() - inicio) / 60,
    }
    np.savez_compressed(salida / "p_ensamble.npz", p=p_ens, y=y, registro=registros)
    (salida / "metricas_anidado.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    registrar(
        f"[s{semilla}] umbral ensamble {umbral:.3f} (antes {umbral_individual:.3f}), "
        f"F1 {resultado['oof_ensamble']['f1']:.4f}"
    )
    evaluar_prueba(salida, conjunto, registrar)  # única evaluación en prueba, umbral nuevo
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config.DIR_CONFIGS / "qcnn" / "base.toml")
    parser.add_argument("--semilla", type=int, default=0)
    parser.add_argument(
        "--resultados", type=Path, default=DIR_QCNN, help="carpeta con cv/ (p. ej. results/cnn_v2)"
    )
    args = parser.parse_args()
    args.resultados.mkdir(parents=True, exist_ok=True)
    ruta_log = args.resultados / f"registro_anidado_s{args.semilla}.log"
    with ruta_log.open("a", encoding="utf-8") as log:

        def registrar(mensaje: str) -> None:
            print(mensaje, flush=True)
            log.write(f"{mensaje}\n")
            log.flush()

        ejecutar(args.config, args.semilla, registrar, args.resultados)
        registrar("Listo.")


if __name__ == "__main__":
    main()
