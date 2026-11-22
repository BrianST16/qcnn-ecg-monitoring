"""CLI: copia a modelos_entrenados/ el ensamble con umbral de CV anidada y escribe su ficha.

Uso: uv run python -m qcnn_ecg.desplegar cnn
     uv run python -m qcnn_ecg.desplegar qcnn

Elige la semilla con mayor F1 del ensamble fuera de pliegue (CV anidada), nunca por la prueba.
La ficha (.json junto al .pt) es lo que el backend expone en GET /api/v1/modelos.
"""

import argparse
import json
import shutil

from qcnn_ecg import config

ORIGEN = {  # clave -> (carpeta con anidado/, nombre de la variante, archivo destino, id, nombre)
    "cnn": ("cnn_v2", "base_rr", "cnn1d_final", "cnn1d-rr-v2", "CNN 1-D"),
    "qcnn": ("qcnn", "qcnn_base", "qcnn_final", "qcnn-hibrida-v1", "QCNN (simulada)"),
}


def desplegar(clave: str) -> dict:
    carpeta, variante, destino, id_modelo, nombre = ORIGEN[clave]
    base = config.DIR_RESULTADOS / carpeta / "anidado"
    candidatas = sorted(base.glob(f"{variante}_s*/metricas_anidado.json"))
    if not candidatas:
        raise SystemExit(f"No hay CV anidada en {base}")
    fichas = [json.loads(r.read_text(encoding="utf-8")) for r in candidatas]
    elegida = max(fichas, key=lambda f: f["oof_ensamble"]["f1"])
    directorio = base / f"{variante}_s{elegida['semilla']}"
    prueba = json.loads((directorio / "metricas_prueba.json").read_text(encoding="utf-8"))["prueba"]

    ficha = {
        "id": id_modelo,
        "nombre": nombre,
        "variante": variante,
        "semilla": elegida["semilla"],
        "criterio": "mayor F1 del ensamble fuera de pliegue (CV anidada) entre semillas",
        "umbral": elegida["umbral_ensamble_anidado"],
        "entrenado_con": "MIT-BIH, DS1 (22 pacientes); ensamble de 5, umbral por CV anidada",
        "metricas_prueba": {
            "f1": round(prueba["f1"], 4),
            "exactitud": round(prueba["exactitud"], 4),
            "auc_roc": round(prueba["auc_roc"], 4),
            "pacientes": 21,
        },
        "oof_ensamble": elegida["oof_ensamble"],
        "prueba": prueba,
        "conjunto": config.NOMBRE_CONJUNTO,
    }
    if clave == "qcnn":  # costo del circuito, calculado por qcnn_ecg.articulo
        insumos = config.DIR_RESULTADOS / "articulo" / "insumos_articulo.json"
        costo = json.loads(insumos.read_text(encoding="utf-8"))["costo_circuito"]
        ficha["circuito"] = {
            "parametros_cuanticos": costo["parametros_circuito"],
            **{
                k: costo["logico"][k]
                for k in ("ancho", "profundidad", "compuertas_1q", "compuertas_2q")
            },
            "compuertas_2q_conectividad_lineal": costo["transpilado_linea"]["compuertas_2q"],
        }

    config.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(directorio / "modelo.pt", config.DIR_MODELOS / f"{destino}.pt")
    (config.DIR_MODELOS / f"{destino}.json").write_text(
        json.dumps(ficha, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{clave}: semilla {ficha['semilla']}, umbral {ficha['umbral']:.3f} -> {destino}.pt")
    return ficha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("claves", nargs="+", choices=list(ORIGEN))
    for clave in parser.parse_args().claves:
        desplegar(clave)


if __name__ == "__main__":
    main()
