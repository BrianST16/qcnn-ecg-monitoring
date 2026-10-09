"""CLI: ablación con validación cruzada por paciente y modelo final con varias semillas.

Uso: uv run python -m qcnn_ecg.experimentos
     uv run python -m qcnn_ecg.experimentos --reanudar --presupuesto-min 38

1. Ablación: cada variante de configs/cnn/ pasa por validación cruzada de 5 pliegues
   de pacientes en DS1 (semilla 0). Se elige la de mayor F1 fuera de pliegue (OOF).
   Una variante se omite si no cabe en LIMITE_ABLACION_MIN.
2. Final: la variante elegida se entrena con todo DS1 (épocas y umbral fijados por
   su CV) con las semillas 0, 1 y 2; si no caben en PRESUPUESTO_MIN se usan menos.
3. Cada modelo final se evalúa UNA vez en prueba (DS2 sin 202, y aparte con 202).
4. Escribe results/cnn/resumen.{csv,md} y copia el modelo de la semilla 0 a
   modelos_entrenados/.
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
from qcnn_ecg.validacion_cruzada import DIR_CV, DIR_FINAL, ejecutar_cv, entrenar_final

VARIANTES = ("base", "kernels_grandes", "flatten", "filtros_x2", "kiranyaz", "dropout_05")
SEMILLAS_FINALES = (0, 1, 2)
# El presupuesto total era de 150 min; unos 27 se usaron en la corrida con validación
# de un solo pliegue (results/cnn_pliegue_unico/) y en dos arranques detenidos.
PRESUPUESTO_MIN = 120.0
LIMITE_ABLACION_MIN = 105.0
METRICAS = ("exactitud", "precision", "sensibilidad", "especificidad")
METRICAS += ("f1", "f1_macro", "auc_roc", "auc_pr")


class Registro:
    """Imprime y guarda en results/cnn/registro_experimentos.log."""

    def __init__(self, ruta: Path, anexar: bool = False):
        self.archivo = ruta.open("a" if anexar else "w", encoding="utf-8")

    def __call__(self, mensaje: str) -> None:
        print(mensaje, flush=True)
        self.archivo.write(mensaje + "\n")
        self.archivo.flush()


def _media_desv(valores, decimales: int = 4) -> str:
    valores = np.asarray(valores, dtype=float)
    desv = valores.std(ddof=1) if len(valores) > 1 else 0.0
    return f"{valores.mean():.{decimales}f} ± {desv:.{decimales}f}"


def _ruta_config(variante: str) -> Path:
    return config.DIR_CONFIGS / "cnn" / f"{variante}.toml"


def _escribir_resumen(ablacion, omitidas, mejor, finales, pruebas, motivo, minutos):
    filas = []
    for r in ablacion:
        fila = {
            "fase": "cv",
            "variante": r["nombre"],
            "semilla": r["semilla"],
            "parametros": r["parametros"],
            "minutos": r["minutos"],
            "epocas_final": r["epocas_final"],
            "umbral": r["umbral_oof"],
        }
        fila |= {f"oof_{m}": r["oof"][m] for m in METRICAS}
        filas.append(fila)
    for f in finales:
        prueba = pruebas[f["directorio"]]
        fila = {
            "fase": "final",
            "variante": f["nombre"],
            "semilla": f["semilla"],
            "parametros": f["parametros"],
            "minutos": f["segundos_entrenamiento"] / 60,
            "epocas_final": f["epocas"],
            "umbral": f["umbral"],
        }
        fila |= {f"prueba_{m}": prueba["prueba"][m] for m in METRICAS}
        fila |= {f"prueba202_{m}": prueba["prueba_con_202"][m] for m in METRICAS}
        fila |= {f"inferencia_{k}": v for k, v in prueba["inferencia"].items()}
        filas.append(fila)
    pd.DataFrame(filas).to_csv(DIR_CORRIDAS / "resumen.csv", index=False)

    lineas = [
        "# Resultados de la CNN 1-D (Ciclo 2)",
        "",
        f"- Fecha: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Equipo: {platform.processor() or platform.machine()}, {torch.get_num_threads()} hilos,"
        f" PyTorch {torch.__version__} (CPU)",
        f"- Conjunto: `{config.NOMBRE_CONJUNTO}` (ver `data/manifiestos/`)",
        f"- Tiempo de cómputo de este experimento: **{minutos:.1f} min** (unos 27 min"
        " adicionales se usaron en la corrida previa con un solo pliegue; presupuesto total"
        " 150 min)",
        f"- Semillas finales: **{', '.join(str(f['semilla']) for f in finales)}**. {motivo}",
        "",
        "## Protocolo",
        "",
        "1. Validación cruzada de 5 pliegues por paciente en DS1 (22 registros), semilla 0.",
        "   Cada pliegue usa parada temprana con el F1 suavizado (media de 3 épocas).",
        "2. Las predicciones fuera de pliegue (OOF) de los 22 pacientes miden cada variante y",
        "   fijan el umbral que maximiza el F1 de la clase anormal.",
        "3. El modelo final se entrena con todo DS1 durante la mediana de las mejores épocas de",
        "   los pliegues, con varias semillas, y se evalúa una sola vez en prueba.",
        "",
        "Pliegues de pacientes en DS1:",
        "",
    ]
    for p in mejor["pliegues"]:
        lineas.append(
            f"- Pliegue {p['pliegue']}: {', '.join(p['registros'])} ({p['latidos']:,} latidos)"
        )

    lineas += [
        "",
        "## Ablación (validación cruzada en DS1, semilla 0)",
        "",
        "| Variante | Parámetros | Min | Épocas final | F1 OOF | Sensibilidad | Especificidad"
        " | AUC-PR OOF | F1 por pliegue |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(ablacion, key=lambda r: -r["oof"]["f1"]):
        o = r["oof"]
        marca = " **(elegida)**" if r["nombre"] == mejor["nombre"] else ""
        lineas.append(
            f"| {r['nombre']}{marca} | {r['parametros']:,} | {r['minutos']:.1f}"
            f" | {r['epocas_final']} | {o['f1']:.4f} | {o['sensibilidad']:.4f}"
            f" | {o['especificidad']:.4f} | {o['auc_pr']:.4f}"
            f" | {_media_desv([p['f1_umbral_propio'] for p in r['pliegues']], 3)} |"
        )
    if omitidas:
        lineas += ["", f"Variantes omitidas por presupuesto: {', '.join(omitidas)}."]

    resultados = [pruebas[f["directorio"]] for f in finales]
    lineas += [
        "",
        f"## Prueba: variante `{mejor['nombre']}`, media ± desviación de {len(finales)} semillas",
        "",
        f"Umbral fijado con la CV en DS1: **{mejor['umbral_oof']:.4f}**. Épocas del modelo final:"
        f" **{mejor['epocas_final']}**. La prueba se evaluó una sola vez por semilla.",
        "",
        "| Métrica | DS2 sin 202 (principal) | DS2 con 202 (comparable con la literatura) |",
        "| :--- | ---: | ---: |",
    ]
    for m in METRICAS:
        lineas.append(
            f"| {m} | {_media_desv([p['prueba'][m] for p in resultados])}"
            f" | {_media_desv([p['prueba_con_202'][m] for p in resultados])} |"
        )
    lineas += ["", "### Matriz de confusión en prueba por semilla ([[VN, FP], [FN, VP]])", ""]
    for f, p in zip(finales, resultados, strict=True):
        lineas.append(f"- Semilla {f['semilla']}: `{p['prueba']['matriz_confusion']}`")

    lineas += [
        "",
        "### Acierto por tipo de latido",
        "",
        "Para `N` equivale a la especificidad; para los demás, a su sensibilidad.",
        "",
        "| Símbolo | Latidos en prueba | Acierto en prueba | Acierto OOF en DS1 |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for s, d in resultados[0]["prueba_por_simbolo"].items():
        aciertos = [p["prueba_por_simbolo"][s]["acierto"] for p in resultados]
        oof = mejor["oof_por_simbolo"].get(s)
        texto_oof = f"{oof['acierto']:.4f}" if oof else "—"
        lineas.append(f"| {s} | {d['n']:,} | {_media_desv(aciertos)} | {texto_oof} |")

    inferencias = [p["inferencia"] for p in resultados]
    lineas += [
        "",
        "### Costo",
        "",
        f"- Parámetros: {finales[0]['parametros']:,}",
        "- Entrenamiento final por semilla: "
        + _media_desv([f["segundos_entrenamiento"] / 60 for f in finales], 2)
        + " min",
        "- Latencia por latido (lote de 1): mediana "
        + _media_desv([i["latencia_ms_mediana"] for i in inferencias], 3)
        + " ms; p95 "
        + _media_desv([i["latencia_ms_p95"] for i in inferencias], 3)
        + " ms",
        "- Throughput (lotes de 256): "
        + _media_desv([i["throughput_latidos_s"] for i in inferencias], 0)
        + " latidos/s",
        "",
        "Modelo para el prototipo: semilla 0 (por convención; las semillas finales no tienen"
        " validación propia para elegir entre ellas), copiado a"
        " `modelos_entrenados/cnn1d_final.pt`.",
        "",
        "## Limitaciones conocidas (pendientes para el Ciclo 3)",
        "",
        "1. **Media móvil no centrada.** La época se elige con la media del F1 de las épocas"
        " t-2, t-1 y t, pero se guarda el estado de la época t: las épocas anteriores le dan"
        " crédito. En 4 de 5 pliegues de `base` la mejor época resultó ser la 3, por lo que el"
        " modelo final entrena solo 3 épocas. Corrección: media centrada (t-1, t, t+1).",
        "2. **Umbral calculado con otros modelos.** El umbral sale de las predicciones de los 5"
        " modelos de la validación cruzada (con su propio punto de parada y reducción de tasa de"
        " aprendizaje), pero se aplica a un modelo final distinto, entrenado con todo DS1 y otra"
        " semilla. Por eso la sensibilidad en prueba varía mucho entre semillas (0.42 a 0.86)."
        " Corrección: usar como modelo final el ensamble de los 5 modelos de la CV.",
        "3. La CV de la ablación usa una sola semilla; el filtro `filtfilt` no es causal.",
        "",
        "La corrida anterior con un único pliegue de validación (5 pacientes) está en"
        " `results/cnn_pliegue_unico/`: allí el umbral elegido (≈ 0.997) no se trasladó a la"
        " prueba (F1 0.29 con AUC-ROC 0.90), lo que motivó este protocolo.",
        "",
    ]
    (DIR_CORRIDAS / "resumen.md").write_text("\n".join(lineas), encoding="utf-8")


def _cargar_cv(variante: str) -> dict | None:
    """Resultado de una validación cruzada ya guardada (modo --reanudar)."""
    ruta = DIR_CV / f"{variante}_s0" / "metricas_cv.json"
    if not ruta.exists():
        return None
    resultado = json.loads(ruta.read_text(encoding="utf-8"))
    resultado["directorio"] = str(ruta.parent)
    return resultado


def _cargar_final(variante: str, semilla: int, epocas: int, umbral: float):
    """(final, prueba) ya guardados si corresponden a la misma variante, épocas y umbral."""
    directorio = DIR_FINAL / f"{variante}_s{semilla}"
    ruta_final, ruta_prueba = directorio / "metricas.json", directorio / "metricas_prueba.json"
    if not (ruta_final.exists() and ruta_prueba.exists()):
        return None
    final = json.loads(ruta_final.read_text(encoding="utf-8"))
    if final["epocas"] != epocas or abs(final["umbral"] - umbral) > 1e-12:
        return None
    final["directorio"] = str(directorio)
    return final, json.loads(ruta_prueba.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--reanudar",
        action="store_true",
        help="reutiliza validaciones cruzadas y modelos finales ya guardados",
    )
    parser.add_argument("--presupuesto-min", type=float, default=PRESUPUESTO_MIN)
    parser.add_argument("--limite-ablacion-min", type=float, default=LIMITE_ABLACION_MIN)
    args = parser.parse_args()
    ejecutar(args.reanudar, args.presupuesto_min, args.limite_ablacion_min)


def ejecutar(reanudar: bool, presupuesto_min: float, limite_ablacion_min: float) -> None:
    DIR_CORRIDAS.mkdir(parents=True, exist_ok=True)
    registrar = Registro(DIR_CORRIDAS / "registro_experimentos.log", anexar=reanudar)
    conjunto = cargar_conjunto()
    inicio = time.perf_counter()
    minutos_previos = 0.0  # cómputo de sesiones anteriores reutilizado al reanudar

    def minutos() -> float:
        return (time.perf_counter() - inicio) / 60

    # 1. Ablación con validación cruzada
    ablacion, nuevas, omitidas = [], [], []
    for variante in VARIANTES:
        guardado = _cargar_cv(variante) if reanudar else None
        if guardado is not None:
            registrar(f"Se reutiliza la validación cruzada de {variante}")
            ablacion.append(guardado)
            minutos_previos += guardado["minutos"]
            continue
        if nuevas:  # la estimación solo usa corridas de esta sesión
            duracion_media = np.mean([r["minutos"] for r in nuevas])
            mejor_hasta_ahora = max(ablacion, key=lambda r: r["oof"]["f1"])
            final_estimada = 0.8 * mejor_hasta_ahora["minutos"]  # 3 entrenamientos en todo DS1
            if minutos() + duracion_media + final_estimada > limite_ablacion_min:
                registrar(f"Se omite {variante}: no cabe en el presupuesto")
                omitidas.append(variante)
                continue
        resultado = ejecutar_cv(_ruta_config(variante), 0, conjunto, registrar)
        ablacion.append(resultado)
        nuevas.append(resultado)
        registrar(f"Tiempo acumulado en esta sesión: {minutos():.1f} min\n")

    mejor = max(ablacion, key=lambda r: r["oof"]["f1"])
    registrar(f"Mejor variante en CV: {mejor['nombre']} (F1 OOF {mejor['oof']['f1']:.4f})")

    # 2. Modelos finales sobre todo DS1; la guarda decide cuántas semillas caben
    finales, pruebas, motivo = [], {}, "Las 3 semillas cupieron en el presupuesto."
    for semilla in SEMILLAS_FINALES:
        guardado = (
            _cargar_final(mejor["nombre"], semilla, mejor["epocas_final"], mejor["umbral_oof"])
            if reanudar
            else None
        )
        if guardado is not None:  # ya entrenado y evaluado en prueba: no se vuelve a evaluar
            final, prueba = guardado
            registrar(f"Se reutiliza el modelo final {mejor['nombre']} semilla {semilla}")
            finales.append(final)
            pruebas[final["directorio"]] = prueba
            minutos_previos += final["segundos_entrenamiento"] / 60
            continue
        if finales:
            duracion = np.mean([f["segundos_entrenamiento"] for f in finales]) / 60
            if minutos() + duracion > presupuesto_min:
                motivo = (
                    f"Solo {len(finales)} semillas: la siguiente no cabía en el tope de"
                    f" {presupuesto_min:.0f} min (desviación respecto al anteproyecto)."
                )
                registrar(motivo)
                break
        final = entrenar_final(
            _ruta_config(mejor["nombre"]),
            semilla,
            mejor["epocas_final"],
            mejor["umbral_oof"],
            conjunto,
            registrar,
        )
        finales.append(final)
        # 3. Prueba: una sola evaluación por modelo final
        pruebas[final["directorio"]] = evaluar_prueba(
            Path(final["directorio"]), conjunto, registrar
        )

    # 4. Modelo para el prototipo y resumen
    elegido = finales[0]
    config.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        Path(elegido["directorio"]) / "modelo.pt", config.DIR_MODELOS / "cnn1d_final.pt"
    )
    (config.DIR_MODELOS / "cnn1d_final.json").write_text(
        json.dumps(
            {
                "variante": elegido["nombre"],
                "semilla": elegido["semilla"],
                "umbral": elegido["umbral"],
                "epocas": elegido["epocas"],
                "parametros": elegido["parametros"],
                "conjunto": config.NOMBRE_CONJUNTO,
                "cv_oof": mejor["oof"],
                "prueba": pruebas[elegido["directorio"]]["prueba"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _escribir_resumen(
        ablacion, omitidas, mejor, finales, pruebas, motivo, minutos_previos + minutos()
    )
    registrar(f"Listo en {minutos():.1f} min. Resumen en {DIR_CORRIDAS / 'resumen.md'}")


if __name__ == "__main__":
    main()
