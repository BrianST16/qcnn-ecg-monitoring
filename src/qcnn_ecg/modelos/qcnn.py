"""Circuito QCNN (Cong, Choi y Lukin, 2019) y su envoltorio PyTorch.

El circuito se define una sola vez con Qiskit. Para entrenar, `QNNTorch` lo simula con tensores
de PyTorch (gradiente exacto por autograd en una pasada); `crear_qnn` (EstimatorQNN de Qiskit,
~170 s por lote de 256 en el backward) queda como referencia para validar el simulador.
No se importa desde `modelos/__init__.py` para que cargar la CNN no cargue qiskit.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from torch import nn

from qcnn_ecg.config import LONGITUD_VENTANA


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
        default_precision=0.0,  # exacto: por defecto Qiskit simula ruido de muestreo (0.0156)
        input_gradients=True,  # el gradiente debe cruzar hacia un front-end clásico previo
    )
    pesos_ini = np.random.default_rng(semilla).normal(0, 0.1, len(pesos))  # evita mesetas
    return TorchConnector(qnn, initial_weights=pesos_ini)


def _pesos_iniciales(n: int, semilla: int) -> np.ndarray:
    return np.random.default_rng(semilla).normal(0, 0.1, n)  # pequeños: evita mesetas


class QNNTorch(nn.Module):
    """Simula en PyTorch el circuito de `circuito_qcnn` (mismo circuito: se lee de Qiskit).

    Entrada (lote, n_qubits) -> <Z> del qubit restante, (lote, 1). Estado exacto, sin ruido.
    """

    def __init__(self, n_qubits: int = 8, semilla: int | None = None):
        super().__init__()
        qc, entradas, pesos, self.salida = circuito_qcnn(n_qubits)
        self.n = n_qubits
        ix = {p: ("x", i) for i, p in enumerate(entradas)}
        ix |= {p: ("w", i) for i, p in enumerate(pesos)}
        self.ops = []  # (puerta, qubits, origen del ángulo)
        for ins in qc.data:
            q = tuple(qc.find_bit(b).index for b in ins.qubits)
            ang = ins.operation.params[0] if ins.operation.params else None
            self.ops.append((ins.operation.name, q, ix[ang] if ang in ix else ang))
        # semilla None: usa la semilla global de torch (así CV y ensamble varían por pliegue)
        n_w = len(pesos)
        w = torch.randn(n_w) * 0.1 if semilla is None else _pesos_iniciales(n_w, semilla)
        self.weight = nn.Parameter(torch.as_tensor(w).float())

    def _eje(self, q: int) -> int:  # qiskit: qubit 0 = bit menos significativo
        return 1 + (self.n - 1 - q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, cd = x.shape[0], torch.complex64
        s = torch.zeros((b,) + (2,) * self.n, dtype=cd)
        s[(slice(None),) + (0,) * self.n] = 1
        for nombre, q, ang in self.ops:
            if nombre == "cx":
                s = self._cx(s, self._eje(q[0]), self._eje(q[1]))
                continue
            if not isinstance(ang, tuple):  # ángulo fijo
                t = x.new_tensor(float(ang))
            else:
                t = x[:, ang[1]] if ang[0] == "x" else self.weight[ang[1]]
            puerta = self._ry if nombre == "ry" else self._rz
            s = puerta(s, t, self._eje(q[0]))
        p = (s.real**2 + s.imag**2).movedim(self._eje(self.salida), -1)
        p = p.reshape(b, -1, 2).sum(1)
        return (p[:, 0] - p[:, 1]).unsqueeze(1)

    @staticmethod
    def _ry(s, t, eje):
        c, sn = torch.cos(t / 2), torch.sin(t / 2)
        u = torch.stack([torch.stack([c, -sn], -1), torch.stack([sn, c], -1)], -2).to(s.dtype)
        forma = s.shape
        s = s.movedim(eje, -1)
        m = s.shape
        s = s.reshape(m[0], -1, 2)
        s = torch.bmm(s, u.transpose(-1, -2)) if u.dim() == 3 else s @ u.T
        return s.reshape(m).movedim(-1, eje).reshape(forma)

    @staticmethod
    def _rz(s, t, eje):  # t escalar (nunca depende del lote en este circuito)
        fase = torch.stack([torch.exp(-0.5j * t), torch.exp(0.5j * t)]).to(s.dtype)
        return (s.movedim(eje, -1) * fase).movedim(-1, eje)

    @staticmethod
    def _cx(s, control, objetivo):
        m = s.movedim((control, objetivo), (-2, -1))
        m = torch.stack([m[..., 0, :], m[..., 1, :].flip(-1)], dim=-2)
        return m.movedim((-2, -1), (control, objetivo))


@dataclass(frozen=True)
class ConfigQCNN:
    n_qubits: int = 8
    filtros: tuple[int, int] = (8, 16)  # front-end clásico: 2 Conv1d
    kernels: tuple[int, int] = (7, 5)
    longitud_entrada: int = LONGITUD_VENTANA
    rasgos_extra: int = 0  # rasgos RR: entran por una capa lineal sumada al logit

    @classmethod
    def desde_dict(cls, datos: dict) -> "ConfigQCNN":
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in datos.items()})

    def a_dict(self) -> dict:
        return asdict(self)


class QCNN(nn.Module):
    """Front-end convolucional -> n_qubits ángulos en [0, pi] -> circuito QCNN -> logit.

    Misma interfaz que CNN1D: forward(x (lote, 1, 288), rr opcional) -> logits (lote,).
    """

    def __init__(self, config: ConfigQCNN | None = None):
        super().__init__()
        self.config = c = config if config is not None else ConfigQCNN()
        f1, f2 = c.filtros
        self.frontal = nn.Sequential(
            nn.Conv1d(1, f1, c.kernels[0], padding=c.kernels[0] // 2),
            nn.BatchNorm1d(f1),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(f1, f2, c.kernels[1], padding=c.kernels[1] // 2),
            nn.BatchNorm1d(f2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(f2, c.n_qubits),
        )
        self.circuito = QNNTorch(c.n_qubits)
        self.escala = nn.Parameter(torch.tensor(1.0))  # <Z> in [-1, 1] -> logit
        self.sesgo = nn.Parameter(torch.tensor(0.0))
        self.rr = nn.Linear(c.rasgos_extra, 1, bias=False) if c.rasgos_extra else None

    def parametros_por_tipo(self) -> dict[str, int]:
        cuanticos = self.circuito.weight.numel()
        total = sum(p.numel() for p in self.parameters())
        return {"clasicos": total - cuanticos, "cuanticos": cuanticos}

    def forward(self, x: torch.Tensor, rr: torch.Tensor | None = None) -> torch.Tensor:
        angulos = torch.pi * torch.sigmoid(self.frontal(x))
        logit = self.escala * self.circuito(angulos).squeeze(-1) + self.sesgo
        if self.rr is not None:
            if rr is None:
                raise ValueError("Este modelo necesita los rasgos RR (rr)")
            logit = logit + self.rr(rr).squeeze(-1)
        return logit
