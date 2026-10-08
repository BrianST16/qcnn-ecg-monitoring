"""CLI: descarga MIT-BIH (si falta) y construye el conjunto procesado.

Uso: uv run python -m qcnn_ecg.preparar_datos
"""

import argparse

from qcnn_ecg import config
from qcnn_ecg.conjunto import construir_conjunto
from qcnn_ecg.datos import descargar_mitdb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir-mitdb", default=config.DIR_MITDB)
    args = parser.parse_args()

    descargar_mitdb(args.dir_mitdb)
    manifiesto = construir_conjunto(args.dir_mitdb)

    print(f"Conjunto {manifiesto['nombre']}: X {tuple(manifiesto['forma_X'])}")
    print(f"Validación: {', '.join(manifiesto['registros_validacion'])}")
    print(f"{'partición':<16}{'registros':>10}{'latidos':>10}{'anormal':>10}{'%':>8}")
    for p, c in manifiesto["conteos"].items():
        n_reg = len(manifiesto["registros_por_particion"][p])
        pct = 100 * c["anormal"] / c["latidos"] if c["latidos"] else 0.0
        print(f"{p:<16}{n_reg:>10}{c['latidos']:>10}{c['anormal']:>10}{pct:>7.1f}%")
    print(f"Manifiesto: {config.DIR_MANIFIESTOS / (config.NOMBRE_CONJUNTO + '.json')}")


if __name__ == "__main__":
    main()
