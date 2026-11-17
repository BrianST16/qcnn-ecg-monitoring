"""CLI: validación cruzada por paciente + ensamble final, para base y base_rr con varias semillas.

Uso: uv run python -m qcnn_ecg.experimentos
     uv run python -m qcnn_ecg.experimentos --variantes base_rr --semillas 0
     uv run python -m qcnn_ecg.experimentos --reanudar

Para cada (variante, semilla): CV de 5 pliegues por paciente en DS1 (modelos de pliegue +
umbral OOF) y una sola evaluación en prueba del ENSAMBLE de los 5 modelos con ese umbral.
La variante se elige por el F1 OOF medio (nunca por la prueba). El ensamble de su semilla 0
se copia a modelos_entrenados/. Escribe results/cnn_v2/resumen.{csv,md}.
"""

import argparse
import json
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qcnn_ecg import config
from qcnn_ecg.conjunto import cargar_conjunto
from qcnn_ecg.entrenar import DIR_CORRIDAS, evaluar_prueba
from qcnn_ecg.validacion_cruzada import DIR_CV, ejecutar_cv

VARIANTES = ("base", "base_rr")
SEMILLAS = (0, 1, 2)
METRICAS = ("exactitud", "precision", "sensibilidad", "especificidad")
METRICAS += ("f1", "f1_macro", "auc_roc", "auc_pr")
# CNN v1 (results/cnn/resumen.md): prueba DS2 sin 202, media de 3 semillas; acierto en latidos A
V1 = {"f1": 0.6058, "auc_roc": 0.8397, "auc_pr": 0.6780, "sensibilidad": 0.6097, "A": 0.3002}


def _media_desv(valores, decimales: int = 4) -> str:
    valores = np.asarray(valores, dtype=float)
    desv = valores.std(ddof=1) if len(valores) > 1 else 0.0
    return f"{valores.mean():.{decimales}f} ± {desv:.{decimales}f}"


def _leer(ruta: Path) -> dict | None:
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def _correr(variante: str, semilla: int, conjunto, reanudar: bool, registrar) -> tuple[dict, dict]:
    """CV + prueba de una (variante, semilla); con `reanudar` reutiliza lo ya guardado."""
    directorio = DIR_CV / f"{variante}_s{semilla}"
    cv = _leer(directorio / "metricas_cv.json") if reanudar else None
    if cv is None or not (directorio / "modelo.pt").exists():
        ruta_config = config.DIR_CONFIGS / "cnn" / f"{variante}.toml"
        cv = ejecutar_cv(ruta_config, semilla, conjunto, registrar)
    else:
        cv["directorio"] = str(directorio)
        registrar(f"Se reutiliza la CV de {variante} semilla {semilla}")
    prueba = _leer(directorio / "metricas_prueba.json") if reanudar else None
    if prueba is None:
        prueba = evaluar_prueba(directorio, conjunto, registrar)
    return cv, prueba


def _escribir_resumen(resultados: dict, elegida: str, minutos: float) -> None:
    filas = []
    for (variante, semilla), (cv, prueba) in resultados.items():
        fila = {"variante": variante, "semilla": semilla, "umbral": cv["umbral_oof"]}
        fila |= {f"oof_{m}": cv["oof"][m] for m in METRICAS}
        fila |= {f"prueba_{m}": prueba["prueba"][m] for m in METRICAS}
        fila["acierto_A"] = prueba["prueba_por_simbolo"].get("A", {}).get("acierto", np.nan)
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(DIR_CORRIDAS / "resumen.csv", index=False)

    lineas = [
        "# Resultados de la CNN 1-D v2 (rasgos RR, media centrada y ensamble)",
        "",
        f"- Fecha: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Equipo: {platform.processor()}, PyTorch {torch.__version__}",
        f"- Conjunto: `{config.NOMBRE_CONJUNTO}` (ver `data/manifiestos/`)",
        f"- Cómputo de esta sesión: **{minutos:.1f} min**. Semillas: {', '.join(map(str, SEMILLAS))}.",
        "",
        "## Protocolo",
        "",
        "1. Validación cruzada de 5 pliegues por paciente en DS1; la parada temprana usa el F1"
        " suavizado con media **centrada** (t-1, t, t+1).",
        "2. Las predicciones fuera de pliegue (OOF) fijan el umbral que maximiza el F1.",
        "3. El modelo final es el **ensamble de los 5 modelos de pliegue** con ese umbral (el"
        " umbral se calculó con los mismos modelos a los que se aplica). Una sola evaluación"
        " en prueba.",
        f"4. Variante elegida por F1 OOF medio: **{elegida}**.",
        "",
        "## Comparación (media ± desviación de las semillas)",
        "",
        "| Variante | F1 OOF | F1 prueba | AUC-ROC prueba | AUC-PR prueba | Sensibilidad prueba |"
        " Acierto en A |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variante, g in tabla.groupby("variante", sort=False):
        marca = " **(elegida)**" if variante == elegida else ""
        lineas.append(
            f"| {variante}{marca} | {_media_desv(g.oof_f1)} | {_media_desv(g.prueba_f1)} |"
            f" {_media_desv(g.prueba_auc_roc)} | {_media_desv(g.prueba_auc_pr)} |"
            f" {_media_desv(g.prueba_sensibilidad)} | {_media_desv(g.acierto_A)} |"
        )
    lineas += [
        f"| v1 `base` (results/cnn/) | 0.7320 | {V1['f1']:.4f} | {V1['auc_roc']:.4f} |"
        f" {V1['auc_pr']:.4f} | {V1['sensibilidad']:.4f} | {V1['A']:.4f} |",
        "",
        "Prueba = DS2 sin el registro 202. La fila v1 es la de `results/cnn/resumen.md`.",
        "",
        "## Detalle por semilla",
        "",
        tabla.round(4).to_csv(index=False),
        "",
    ]
    (DIR_CORRIDAS / "resumen.md").write_text("\n".join(lineas), encoding="utf-8")


def ejecutar(variantes, semillas, reanudar: bool) -> None:
    DIR_CORRIDAS.mkdir(parents=True, exist_ok=True)
    ruta_log = DIR_CORRIDAS / "registro_experimentos.log"
    log = ruta_log.open("a" if reanudar else "w", encoding="utf-8")

    def registrar(mensaje: str) -> None:
        print(mensaje, flush=True)
        log.write(mensaje + "\n")
        log.flush()

    conjunto, inicio, resultados = cargar_conjunto(), time.perf_counter(), {}
    for variante in variantes:
        for semilla in semillas:
            resultados[(variante, semilla)] = _correr(
                variante, semilla, conjunto, reanudar, registrar
            )
            registrar(f"Tiempo acumulado: {(time.perf_counter() - inicio) / 60:.1f} min\n")

    f1_oof = {v: np.mean([resultados[(v, s)][0]["oof"]["f1"] for s in semillas]) for v in variantes}
    elegida = max(f1_oof, key=f1_oof.get)
    registrar(f"Variante elegida por F1 OOF medio: {elegida} ({f1_oof[elegida]:.4f})")

    cv, prueba = resultados[(elegida, semillas[0])]
    config.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(cv["directorio"]) / "modelo.pt", config.DIR_MODELOS / "cnn1d_final.pt")
    (config.DIR_MODELOS / "cnn1d_final.json").write_text(
        json.dumps(
            {
                "variante": elegida,
                "semilla": semillas[0],
                "ensamble_de": 5,
                "umbral": cv["umbral_oof"],
                "parametros_por_modelo": cv["parametros"],
                "conjunto": config.NOMBRE_CONJUNTO,
                "cv_oof": cv["oof"],
                "prueba": prueba["prueba"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _escribir_resumen(resultados, elegida, (time.perf_counter() - inicio) / 60)
    registrar(f"Listo. Resumen en {DIR_CORRIDAS / 'resumen.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variantes", nargs="+", default=list(VARIANTES))
    parser.add_argument("--semillas", nargs="+", type=int, default=list(SEMILLAS))
    parser.add_argument("--reanudar", action="store_true", help="reutiliza CV y pruebas ya hechas")
    args = parser.parse_args()
    ejecutar(args.variantes, args.semillas, args.reanudar)


if __name__ == "__main__":
    main()
