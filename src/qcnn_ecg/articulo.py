"""CLI: insumos del artículo (comparación CNN v2 / QCNN en prueba, costo del circuito y figuras).

Uso: uv run --with pylatexenc python -m qcnn_ecg.articulo
(`pylatexenc` solo hace falta para dibujar el circuito; sin él se omite esa figura.)

- Predice la prueba (DS2 sin 202) con el ensamble de cada (modelo, semilla) ya entrenado.
- Bootstrap pareado por paciente (se remuestrean los 21 pacientes de prueba) de la diferencia
  QCNN - CNN en AUC-ROC, AUC-PR y F1, promediada entre las 3 semillas.
- Costo del circuito: ancho, profundidad y compuertas de 1 y 2 qubits (lógico y transpilado).
- Figuras en docs/articulo/figuras/ y números en results/articulo/.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from qiskit import transpile  # noqa: E402
from qiskit.transpiler import CouplingMap  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from qcnn_ecg import config  # noqa: E402
from qcnn_ecg.conjunto import cargar_conjunto  # noqa: E402
from qcnn_ecg.entrenar import cargar_modelo  # noqa: E402
from qcnn_ecg.evaluacion import predecir_probabilidades  # noqa: E402
from qcnn_ecg.modelos.qcnn import circuito_qcnn  # noqa: E402

SALIDA = config.DIR_RESULTADOS / "articulo"
FIGURAS = config.RAIZ / "docs" / "articulo" / "figuras"
SEMILLAS = (0, 1, 2)
MODELOS = {  # nombre -> (carpeta de la CV, color categórico en orden fijo)
    "CNN v2 base": (config.DIR_RESULTADOS / "cnn_v2" / "cv" / "base_s{}", "#2a78d6"),
    "CNN v2 base_rr": (config.DIR_RESULTADOS / "cnn_v2" / "cv" / "base_rr_s{}", "#eb6834"),
    "QCNN": (config.DIR_RESULTADOS / "qcnn" / "cv" / "qcnn_base_s{}", "#1baf7a"),
}
N_BOOTSTRAP = 2000


def predicciones() -> dict:
    """{(modelo, semilla): (p, umbral)} en prueba; se guardan para no repetir la inferencia."""
    cache = SALIDA / "predicciones_prueba.npz"
    conjunto = cargar_conjunto()
    m = conjunto.mascara("prueba")
    y, registro = conjunto.y[m], conjunto.registro[m]
    if cache.exists():
        datos = np.load(cache)
        res = {
            (nombre, s): (datos[f"{nombre}|{s}"], float(datos[f"{nombre}|{s}|umbral"]))
            for nombre in MODELOS
            for s in SEMILLAS
        }
        return y, registro, res
    res = {}
    for nombre, (patron, _) in MODELOS.items():
        for s in SEMILLAS:
            modelo, umbral = cargar_modelo(patron.with_name(patron.name.format(s)) / "modelo.pt")
            p = predecir_probabilidades(modelo, conjunto.X[m], rr=conjunto.rr[m])
            res[(nombre, s)] = (p, umbral)
            print(f"predicho {nombre} s{s}", flush=True)
    planos = {f"{k[0]}|{k[1]}": v[0] for k, v in res.items()}
    planos |= {f"{k[0]}|{k[1]}|umbral": np.array(v[1]) for k, v in res.items()}
    np.savez_compressed(cache, **planos)
    return y, registro, res


def _metricas(y, p, umbral):
    return (
        roc_auc_score(y, p),
        average_precision_score(y, p),
        f1_score(y, (p >= umbral).astype(int), zero_division=0),
    )


def bootstrap(y, registro, res, referencia: str) -> dict:
    """IC 95 % de QCNN - referencia (media entre semillas), remuestreando pacientes."""
    pacientes = np.unique(registro)
    indices = {r: np.flatnonzero(registro == r) for r in pacientes}
    rng = np.random.default_rng(0)
    def delta(idx) -> np.ndarray:
        d = []
        for s in SEMILLAS:
            (pq, uq), (pr, ur) = res[("QCNN", s)], res[(referencia, s)]
            d.append(np.subtract(_metricas(y[idx], pq[idx], uq), _metricas(y[idx], pr[idx], ur)))
        return np.mean(d, axis=0)

    deltas = []
    for _ in range(N_BOOTSTRAP):
        idx = np.concatenate([indices[r] for r in rng.choice(pacientes, len(pacientes))])
        if len(np.unique(y[idx])) == 2:
            deltas.append(delta(idx))
    deltas = np.array(deltas)
    observado = delta(np.arange(len(y)))
    salida = {}
    for j, nombre in enumerate(("auc_roc", "auc_pr", "f1")):
        bajo, alto = np.percentile(deltas[:, j], [2.5, 97.5])
        salida[nombre] = {
            "delta": float(observado[j]),
            "ic95": [float(bajo), float(alto)],
            "p_bilateral": float(2 * min((deltas[:, j] <= 0).mean(), (deltas[:, j] >= 0).mean())),
        }
    return salida


def figura_roc_pr(y, res) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for nombre, (_, color) in MODELOS.items():
        for k, s in enumerate(SEMILLAS):
            p = res[(nombre, s)][0]
            fpr, tpr, _ = roc_curve(y, p)
            prec, rec, _ = precision_recall_curve(y, p)
            etiqueta = nombre if k == 0 else None
            a.plot(fpr, tpr, color=color, lw=1.5, alpha=0.85, label=etiqueta)
            b.plot(rec, prec, color=color, lw=1.5, alpha=0.85, label=etiqueta)
    a.plot([0, 1], [0, 1], color="#9a9a94", lw=1, ls="--")
    b.axhline(y.mean(), color="#9a9a94", lw=1, ls="--")
    a.set(xlabel="Tasa de falsos positivos", ylabel="Sensibilidad", title="(a) Curva ROC")
    b.set(xlabel="Sensibilidad (recall)", ylabel="Precisión", title="(b) Curva precisión-recall")
    for ax in (a, b):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.grid(color="#e6e5e0", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    b.legend(loc="lower left", frameon=False, title="Tres semillas por modelo")
    fig.tight_layout()
    fig.savefig(FIGURAS / "roc_pr_prueba.png", dpi=200)
    plt.close(fig)


def costo_circuito() -> dict:
    qc, _, pesos, salida = circuito_qcnn(8)

    def contar(c) -> dict:
        n2 = sum(1 for i in c.data if i.operation.num_qubits == 2)
        return {
            "ancho": c.num_qubits,
            "profundidad": c.depth(),
            "profundidad_2q": c.depth(lambda i: i.operation.num_qubits == 2),
            "compuertas_1q": len(c.data) - n2,
            "compuertas_2q": n2,
            "operaciones": dict(c.count_ops()),
        }

    base = ["rz", "sx", "x", "cx"]
    res = {
        "parametros_circuito": len(pesos),
        "qubit_lectura": salida,
        "logico": contar(qc),
        "transpilado_todos_con_todos": contar(
            transpile(qc, basis_gates=base, optimization_level=1, seed_transpiler=0)
        ),
        "transpilado_linea": contar(
            transpile(qc, basis_gates=base, coupling_map=CouplingMap.from_line(8),
                      optimization_level=1, seed_transpiler=0)
        ),  # fmt: skip
    }
    try:
        fig = qc.draw("mpl", fold=40, scale=0.6)
        fig.savefig(FIGURAS / "circuito_qcnn.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001 - la figura es opcional (falta pylatexenc)
        print(f"No se dibujó el circuito: {e}")
    return res


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    y, registro, res = predicciones()
    resultado = {
        "prueba_por_semilla": {
            f"{n}|{s}": dict(
                zip(("auc_roc", "auc_pr", "f1"), _metricas(y, *res[(n, s)]), strict=True)
            )
            for n in MODELOS
            for s in SEMILLAS
        },
        "bootstrap_qcnn_menos": {
            ref: bootstrap(y, registro, res, ref) for ref in MODELOS if ref != "QCNN"
        },
        "costo_circuito": costo_circuito(),
    }
    figura_roc_pr(y, res)
    (SALIDA / "insumos_articulo.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(resultado["bootstrap_qcnn_menos"], indent=1))


if __name__ == "__main__":
    main()
