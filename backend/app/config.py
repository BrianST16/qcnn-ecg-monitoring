from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from typing import Literal

DIR_BACKEND = Path(__file__).resolve().parents[1]


class Ajustes(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=DIR_BACKEND / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = Field(min_length=20)
    origenes_permitidos: str = "http://localhost:5173"
    docs_habilitados: bool = True
    modelo_simulado: bool = True
    max_analisis_simultaneos: int = Field(default=2, ge=1, le=16)
    limite_peticiones_por_minuto: int = Field(default=30, ge=1, le=1000)
    hosts_permitidos: str = "localhost,127.0.0.1"
    
    ruta_modelo_cnn: Path = DIR_BACKEND.parent / "modelos_entrenados" / "cnn1d_final.pt"
    torch_hilos: int = Field(default=4, ge=1, le=64)
    qcnn_modo: Literal["desactivada", "maqueta", "real"] = "desactivada"
    ruta_modelo_qcnn: Path = DIR_BACKEND.parent / "modelos_entrenados" / "qcnn_final.pt"

    @property
    def lista_origenes(self) -> list[str]:
        return [o.strip() for o in self.origenes_permitidos.split(",") if o.strip()]
    
    @property
    def lista_hosts(self) -> list[str]:
        return [h.strip() for h in self.hosts_permitidos.split(",") if h.strip()]


@lru_cache
def get_ajustes() -> Ajustes:
    return Ajustes()