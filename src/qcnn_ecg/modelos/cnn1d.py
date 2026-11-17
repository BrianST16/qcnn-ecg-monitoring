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
    rasgos_extra: int = 0  # rasgos de ritmo (RR) que se concatenan tras el pooling; 0 = solo señal

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
        self._n_reduccion = len(reduccion)
        self.clasificador = nn.Sequential(
            *reduccion,
            nn.Dropout(config.dropout),
            nn.Linear(entrada_densa + config.rasgos_extra, config.ocultas),
            nn.ReLU(),
            nn.Linear(config.ocultas, 1),
        )

    def forward(self, x: torch.Tensor, rr: torch.Tensor | None = None) -> torch.Tensor:
        """x: (lote, 1, 288) y rr: (lote, rasgos_extra) opcional -> logits (lote,)."""
        k = self._n_reduccion
        h = self.clasificador[:k](self.extractor(x))
        if self.config.rasgos_extra:
            if rr is None:
                raise ValueError("Este modelo necesita los rasgos RR (rr)")
            h = torch.cat([h, rr], dim=1)
        return self.clasificador[k:](h).squeeze(-1)


class Ensamble(nn.Module):
    """Promedia las probabilidades de varios modelos; devuelve su logit (misma interfaz que CNN1D)."""

    def __init__(self, modelos: list[CNN1D]):
        super().__init__()
        self.modelos = nn.ModuleList(modelos)
        self.config = modelos[0].config

    def forward(self, x: torch.Tensor, rr: torch.Tensor | None = None) -> torch.Tensor:
        p = torch.stack([torch.sigmoid(m(x, rr)) for m in self.modelos]).mean(0)
        p = p.clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
