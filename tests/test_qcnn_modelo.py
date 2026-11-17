import pytest
import torch

from qcnn_ecg.entrenar import cargar_modelo, config_desde_dict
from qcnn_ecg.evaluacion import predecir_probabilidades
from qcnn_ecg.modelos import Ensamble
from qcnn_ecg.modelos.qcnn import QCNN, ConfigQCNN


def test_qcnn_forma_rr_y_conteo_de_parametros():
    m = QCNN(ConfigQCNN(rasgos_extra=4))
    assert m(torch.zeros(5, 1, 288), torch.ones(5, 4)).shape == (5,)
    with pytest.raises(ValueError):
        m(torch.zeros(5, 1, 288))
    n = m.parametros_por_tipo()
    assert n["cuanticos"] == 18 and sum(n.values()) == sum(p.numel() for p in m.parameters())


def test_qcnn_el_gradiente_llega_al_frontal_y_al_circuito():
    m = QCNN()
    m(torch.randn(8, 1, 288)).sum().backward()
    assert m.frontal[0].weight.grad.abs().sum() > 0
    assert m.circuito.weight.grad.abs().sum() > 0


def test_qcnn_ensamble_guardar_cargar_predecir(tmp_path):
    cfg = ConfigQCNN(rasgos_extra=4)
    modelos = [QCNN(cfg).eval() for _ in range(2)]
    ruta = tmp_path / "modelo.pt"
    torch.save(
        {
            "estados": [m.state_dict() for m in modelos],
            "config_modelo": cfg.a_dict(),
            "umbral": 0.4,
        },
        ruta,
    )
    cargado, umbral = cargar_modelo(ruta)
    assert isinstance(cargado, Ensamble) and umbral == 0.4
    assert isinstance(config_desde_dict(cfg.a_dict()), ConfigQCNN)
    X, rr = torch.randn(6, 288).numpy(), torch.ones(6, 4).numpy()
    esperado = Ensamble(modelos).eval()
    torch.testing.assert_close(
        torch.tensor(predecir_probabilidades(cargado, X, rr=rr)),
        torch.tensor(predecir_probabilidades(esperado, X, rr=rr)),
    )
