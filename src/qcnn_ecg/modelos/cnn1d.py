"""CNN 1-D compacta para clasificar latidos (entrada de 288 muestras, salida un logit).

Arquitectura: bloques Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d, seguidos de
pooling global ("gap") o aplanado ("flatten"), una capa oculta y la salida.
Inspirada en la CNN compacta de Kiranyaz et al. (2016), con normalización por
lotes como pide el anteproyecto.
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from qcnn_ecg.config import LONGITUD_VENTANA


@dataclass(frozen=True)
class ConfigCNN:
    filtros: tuple[int, ...] = (16, 32, 64)
    kernels: tuple[int, ...] = (7, 5, 3)
    pools: tuple[int, ...] = (2, 2, 2)
    cabeza: str = "gap"  # "gap" (pooling global promedio) o "flatten"
    ocultas: int = 32
    dropout: float = 0.3
    longitud_entrada: int = LONGITUD_VENTANA

    def __post_init__(self):
        if not len(self.filtros) == len(self.kernels) == len(self.pools):
            raise ValueError("filtros, kernels y pools deben tener la misma longitud")
        if self.cabeza not in ("gap", "flatten"):
            raise ValueError(f"cabeza desconocida: {self.cabeza}")

    @classmethod
    def desde_dict(cls, datos: dict) -> "ConfigCNN":
        datos = {k: tuple(v) if isinstance(v, list) else v for k, v in datos.items()}
        return cls(**datos)

    def a_dict(self) -> dict:
        return asdict(self)


class CNN1D(nn.Module):
    def __init__(self, config: ConfigCNN | None = None):
        super().__init__()
        config = config if config is not None else ConfigCNN()
        self.config = config
        bloques, canales, longitud = [], 1, config.longitud_entrada
        for filtros, kernel, pool in zip(config.filtros, config.kernels, config.pools, strict=True):
            bloques += [
                nn.Conv1d(canales, filtros, kernel, padding=kernel // 2),
                nn.BatchNorm1d(filtros),
                nn.ReLU(),
                nn.MaxPool1d(pool),
            ]
            canales, longitud = filtros, longitud // pool
        self.extractor = nn.Sequential(*bloques)

        if config.cabeza == "gap":
            entrada_densa = canales
            reduccion = [nn.AdaptiveAvgPool1d(1), nn.Flatten()]
        else:
            entrada_densa = canales * longitud
            reduccion = [nn.Flatten()]
        self.clasificador = nn.Sequential(
            *reduccion,
            nn.Dropout(config.dropout),
            nn.Linear(entrada_densa, config.ocultas),
            nn.ReLU(),
            nn.Linear(config.ocultas, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (lote, 1, 288) -> logits (lote,)."""
        return self.clasificador(self.extractor(x)).squeeze(-1)


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
