"""Figuras de entrenamiento y evaluación (se guardan como PNG)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from qcnn_ecg.config import NOMBRES_CLASES  # noqa: E402


def graficar_curvas(historial: pd.DataFrame, ruta: Path, titulo: str = "") -> None:
    fig, (eje_perdida, eje_f1) = plt.subplots(1, 2, figsize=(12, 4))
    eje_perdida.plot(historial["epoca"], historial["perdida_ent"], label="Entrenamiento")
    eje_perdida.plot(historial["epoca"], historial["perdida_val"], label="Validación")
    eje_perdida.set(title="Pérdida (BCE ponderada)", xlabel="Época", ylabel="Pérdida")
    eje_f1.plot(historial["epoca"], historial["f1_ent_05"], label="Entrenamiento (umbral 0.5)")
    eje_f1.plot(historial["epoca"], historial["f1_val_05"], label="Validación (umbral 0.5)")
    eje_f1.plot(historial["epoca"], historial["f1_val"], label="Validación (umbral óptimo)")
    mejor = historial.loc[historial["f1_val"].idxmax()]
    eje_f1.axvline(mejor["epoca"], color="gray", ls="--", lw=1, label="Mejor época")
    eje_f1.set(title="F1 de la clase ANORMAL", xlabel="Época", ylabel="F1")
    for eje in (eje_perdida, eje_f1):
        eje.grid(alpha=0.3)
        eje.legend()
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def graficar_matriz(matriz: list[list[int]], ruta: Path, titulo: str = "") -> None:
    matriz = np.asarray(matriz)
    porcentaje = matriz / matriz.sum(axis=1, keepdims=True)
    fig, eje = plt.subplots(figsize=(4.8, 4.2))
    eje.imshow(porcentaje, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            color = "white" if porcentaje[i, j] > 0.5 else "black"
            eje.text(
                j,
                i,
                f"{matriz[i, j]:,}\n({porcentaje[i, j]:.1%})",
                ha="center",
                va="center",
                color=color,
            )
    eje.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=NOMBRES_CLASES,
        yticklabels=NOMBRES_CLASES,
        xlabel="Predicción",
        ylabel="Real",
        title=titulo,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def graficar_roc_pr(y: np.ndarray, p: np.ndarray, ruta: Path, titulo: str = "") -> None:
    fpr, tpr, _ = roc_curve(y, p)
    precision, sensibilidad, _ = precision_recall_curve(y, p)
    fig, (eje_roc, eje_pr) = plt.subplots(1, 2, figsize=(10, 4))
    eje_roc.plot(fpr, tpr)
    eje_roc.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
    eje_roc.set(title="Curva ROC", xlabel="1 - especificidad", ylabel="Sensibilidad")
    eje_pr.plot(sensibilidad, precision)
    eje_pr.axhline(y.mean(), color="gray", ls="--", lw=1)
    eje_pr.set(title="Curva precisión-sensibilidad", xlabel="Sensibilidad", ylabel="Precisión")
    for eje in (eje_roc, eje_pr):
        eje.grid(alpha=0.3)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
