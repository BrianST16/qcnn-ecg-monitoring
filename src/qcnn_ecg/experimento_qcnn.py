"""CLI: validación cruzada por paciente + evaluación única en prueba de la QCNN (results/qcnn/).

Uso: uv run python -m qcnn_ecg.experimento_qcnn --semilla 0
Mismo protocolo que la CNN (qcnn_ecg.validacion_cruzada): 5 pliegues por paciente, umbral OOF,
ensamble de los 5 modelos, prueba DS2 una sola vez.
"""

import argparse
from pathlib import Path

from qcnn_ecg import config
from qcnn_ecg.conjunto import cargar_conjunto
from qcnn_ecg.entrenar import evaluar_prueba
from qcnn_ecg.validacion_cruzada import ejecutar_cv

DIR_QCNN = config.DIR_RESULTADOS / "qcnn"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config.DIR_CONFIGS / "qcnn" / "base.toml")
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args()

    DIR_QCNN.mkdir(parents=True, exist_ok=True)
    with (DIR_QCNN / f"registro_s{args.semilla}.log").open("w", encoding="utf-8") as log:

        def registrar(mensaje: str) -> None:
            print(mensaje, flush=True)
            log.write(f"{mensaje}\n")
            log.flush()

        conjunto = cargar_conjunto()
        cv = ejecutar_cv(args.config, args.semilla, conjunto, registrar, DIR_QCNN / "cv")
        evaluar_prueba(Path(cv["directorio"]), conjunto, registrar)
        registrar("Listo.")


if __name__ == "__main__":
    main()
