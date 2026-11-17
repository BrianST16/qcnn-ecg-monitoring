"""Circuito QCNN (Cong, Choi y Lukin, 2019) y su envoltorio PyTorch.

Hito 1: solo el circuito y la QNN (sin front-end clásico, RR ni logit; eso es del hito 2).
No se importa desde `modelos/__init__.py` para que cargar la CNN no cargue qiskit.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN


def _conv(p) -> QuantumCircuit:
    """Bloque de convolución de 2 qubits, 3 parámetros."""
    c = QuantumCircuit(2)
    c.rz(-np.pi / 2, 1)
    c.cx(1, 0)
    c.rz(p[0], 0)
    c.ry(p[1], 1)
    c.cx(0, 1)
    c.ry(p[2], 1)
    c.cx(1, 0)
    c.rz(np.pi / 2, 0)
    return c


def _pool(p) -> QuantumCircuit:
    """Bloque de pooling de 2 qubits (qubit 0 = descartado, qubit 1 = se conserva)."""
    c = QuantumCircuit(2)
    c.rz(-np.pi / 2, 1)
    c.cx(1, 0)
    c.rz(p[0], 0)
    c.ry(p[1], 1)
    c.cx(0, 1)
    c.ry(p[2], 1)
    return c


def circuito_qcnn(n_qubits: int = 8):
    """Devuelve (circuito, parámetros de entrada, parámetros de pesos).

    RY(x_i) por qubit; luego rondas de convolución (pares pares, pares impares con cierre
    circular) + pooling hasta quedar 1 qubit. Pesos compartidos por capa: 6 por ronda.
    """
    entradas = ParameterVector("x", n_qubits)
    rondas = int(np.log2(n_qubits))
    pesos = ParameterVector("w", 6 * rondas)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(entradas[i], i)
    activos = list(range(n_qubits))
    for r in range(rondas):
        pc, pp = pesos[6 * r : 6 * r + 3], pesos[6 * r + 3 : 6 * r + 6]
        m = len(activos)
        pares = [(activos[i], activos[i + 1]) for i in range(0, m - 1, 2)]
        if m > 2:  # segunda pasada, desplazada y circular
            pares += [(activos[i], activos[(i + 1) % m]) for i in range(1, m, 2)]
        for a, b in pares:
            qc.compose(_conv(pc), [a, b], inplace=True)
        for a, b in zip(activos[0::2], activos[1::2], strict=True):
            qc.compose(_pool(pp), [a, b], inplace=True)
        activos = activos[1::2]
    return qc, entradas, pesos, activos[0]


def crear_qnn(n_qubits: int = 8, semilla: int = 0) -> TorchConnector:
    """QNN como `nn.Module`: entrada (lote, n_qubits) -> <Z> del qubit restante, (lote, 1)."""
    qc, entradas, pesos, salida = circuito_qcnn(n_qubits)
    observable = SparsePauliOp.from_sparse_list([("Z", [salida], 1.0)], num_qubits=n_qubits)
    qnn = EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=list(entradas),
        weight_params=list(pesos),
        estimator=StatevectorEstimator(),
        input_gradients=True,  # el gradiente debe cruzar hacia un front-end clásico previo
    )
    pesos_ini = np.random.default_rng(semilla).normal(0, 0.1, len(pesos))  # evita mesetas
    return TorchConnector(qnn, initial_weights=pesos_ini)
