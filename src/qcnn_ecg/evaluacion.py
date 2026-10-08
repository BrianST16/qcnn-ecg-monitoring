"""Métricas de clasificación, umbral óptimo y tiempo de inferencia.

Se usan las mismas funciones para la CNN y la QCNN, de modo que la comparación
ocurra en condiciones idénticas.
"""

import time

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


@torch.no_grad()
def predecir_logits(
    modelo: torch.nn.Module, X: np.ndarray, tamano_lote: int = 2048
) -> torch.Tensor:
    """Logits de la clase ANORMAL para cada latido (X: (n, 288))."""
    modelo.eval()
    tensor = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).unsqueeze(1)
    return torch.cat([modelo(tensor[i : i + tamano_lote]) for i in range(0, len(X), tamano_lote)])


def predecir_probabilidades(
    modelo: torch.nn.Module, X: np.ndarray, tamano_lote: int = 2048
) -> np.ndarray:
    """Probabilidad de la clase ANORMAL para cada latido."""
    return torch.sigmoid(predecir_logits(modelo, X, tamano_lote)).numpy()


def umbral_optimo_f1(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Umbral que maximiza el F1 de la clase anormal; devuelve (umbral, f1)."""
    precision, sensibilidad, umbrales = precision_recall_curve(y, p)
    precision, sensibilidad = precision[:-1], sensibilidad[:-1]  # alineadas con `umbrales`
    denominador = precision + sensibilidad
    f1 = np.divide(
        2 * precision * sensibilidad,
        denominador,
        out=np.zeros_like(denominador),
        where=denominador > 0,
    )
    mejor = int(np.argmax(f1))
    return float(umbrales[mejor]), float(f1[mejor])


def calcular_metricas(y: np.ndarray, p: np.ndarray, umbral: float = 0.5) -> dict:
    """Métricas del anteproyecto (y algunas adicionales) con la clase anormal como positiva."""
    pred = (p >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    def razon(a, b):
        return float(a / b) if b else 0.0

    return {
        "n": int(len(y)),
        "umbral": float(umbral),
        "exactitud": razon(tp + tn, len(y)),
        "precision": razon(tp, tp + fp),
        "sensibilidad": razon(tp, tp + fn),
        "especificidad": razon(tn, tn + fp),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auc_roc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "auc_pr": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "matriz_confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def acierto_por_simbolo(
    y: np.ndarray, p: np.ndarray, simbolos: np.ndarray, umbral: float
) -> dict[str, dict]:
    """Fracción de latidos bien clasificados por símbolo original.

    Para 'N' equivale a la especificidad; para los anormales, a su sensibilidad.
    """
    acierto = (p >= umbral).astype(int) == y
    resultado = {}
    for s in sorted(set(simbolos.tolist()), key=lambda s: -(simbolos == s).sum()):
        m = simbolos == s
        resultado[s] = {"n": int(m.sum()), "acierto": float(acierto[m].mean())}
    return resultado


@torch.no_grad()
def medir_inferencia(
    modelo: torch.nn.Module,
    X: np.ndarray,
    repeticiones: int = 1000,
    calentamiento: int = 50,
    tamano_lote: int = 256,
) -> dict:
    """Latencia por latido (lote de 1) y throughput (lotes de `tamano_lote`) en CPU."""
    modelo.eval()
    tensor = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).unsqueeze(1)
    for i in range(calentamiento):
        modelo(tensor[i : i + 1])
    tiempos = []
    for i in range(repeticiones):
        inicio = time.perf_counter()
        modelo(tensor[i % len(tensor) : i % len(tensor) + 1])
        tiempos.append(time.perf_counter() - inicio)
    tiempos_ms = np.array(tiempos) * 1000

    n_lotes = max(1, min(len(tensor) // tamano_lote, 40))
    inicio = time.perf_counter()
    for i in range(n_lotes):
        modelo(tensor[i * tamano_lote : (i + 1) * tamano_lote])
    duracion = time.perf_counter() - inicio
    return {
        "latencia_ms_mediana": float(np.median(tiempos_ms)),
        "latencia_ms_p95": float(np.percentile(tiempos_ms, 95)),
        "throughput_latidos_s": float(n_lotes * tamano_lote / duracion),
        "hilos": torch.get_num_threads(),
    }
