"""Bucle de entrenamiento con pesos de clase, parada temprana y selección por F1.

La mejor época se elige por el F1 de la clase anormal en validación, calculado
con el umbral óptimo de esa época (el mismo criterio con el que luego se fija
el umbral definitivo).
"""

import copy
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from qcnn_ecg.conjunto import CargadorLotes
from qcnn_ecg.evaluacion import predecir_logits, umbral_optimo_f1


@dataclass(frozen=True)
class ConfigEntrenamiento:
    tasa_aprendizaje: float = 1e-3
    tamano_lote: int = 256
    epocas_max: int = 30
    paciencia: int = 5
    paciencia_lr: int = 2
    factor_lr: float = 0.1
    hilos: int = 4

    def a_dict(self) -> dict:
        return asdict(self)


def fijar_semillas(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


def peso_positivo(y: np.ndarray) -> float:
    """pos_weight de BCEWithLogitsLoss: n_normales / n_anormales del entrenamiento."""
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def entrenar_modelo(
    modelo: nn.Module,
    X_ent: np.ndarray,
    y_ent: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: ConfigEntrenamiento,
    semilla: int,
    registrar=print,
) -> tuple[nn.Module, list[dict]]:
    """Entrena y devuelve (modelo con el mejor estado, historial por época)."""
    torch.set_num_threads(config.hilos)
    cargador = CargadorLotes(X_ent, y_ent, config.tamano_lote, barajar=True, semilla=semilla)
    criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(peso_positivo(y_ent)))
    optimizador = torch.optim.Adam(modelo.parameters(), lr=config.tasa_aprendizaje)
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador, mode="max", factor=config.factor_lr, patience=config.paciencia_lr
    )
    y_val_t = torch.from_numpy(y_val.astype(np.float32))

    historial, mejor_f1, mejor_estado, sin_mejora = [], -1.0, None, 0
    for epoca in range(1, config.epocas_max + 1):
        inicio = time.perf_counter()
        modelo.train()
        perdida_total, logits_ent, y_vistos = 0.0, [], []
        for xb, yb in cargador:
            optimizador.zero_grad()
            logits = modelo(xb)
            perdida = criterio(logits, yb)
            perdida.backward()
            optimizador.step()
            perdida_total += perdida.item() * len(yb)
            logits_ent.append(logits.detach())
            y_vistos.append(yb)
        y_vistos = torch.cat(y_vistos).numpy().astype(int)
        pred_ent = (torch.cat(logits_ent) >= 0).numpy().astype(int)

        logits_val = predecir_logits(modelo, X_val)
        with torch.no_grad():
            perdida_val = criterio(logits_val, y_val_t).item()
        p_val = torch.sigmoid(logits_val).numpy()
        umbral, f1_val = umbral_optimo_f1(y_val, p_val)

        planificador.step(f1_val)
        fila = {
            "epoca": epoca,
            "perdida_ent": perdida_total / len(y_vistos),
            "f1_ent_05": float(f1_score(y_vistos, pred_ent, zero_division=0)),
            "perdida_val": perdida_val,
            "f1_val_05": float(f1_score(y_val, (p_val >= 0.5).astype(int), zero_division=0)),
            "f1_val": f1_val,
            "umbral_val": umbral,
            "lr": optimizador.param_groups[0]["lr"],
            "segundos": time.perf_counter() - inicio,
        }
        historial.append(fila)
        registrar(
            f"  época {epoca:2d} | pérdida ent {fila['perdida_ent']:.4f} val {perdida_val:.4f}"
            f" | F1 val {f1_val:.4f} (umbral {umbral:.2f}) | {fila['segundos']:.1f} s"
        )

        if f1_val > mejor_f1:
            mejor_f1, mejor_estado, sin_mejora = f1_val, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= config.paciencia:
                registrar(f"  parada temprana en la época {epoca}")
                break
    modelo.load_state_dict(mejor_estado)
    return modelo, historial
