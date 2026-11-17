import torch

from qcnn_ecg.modelos.qcnn import QNNTorch, crear_qnn


def test_simulador_torch_igual_que_qiskit_en_salida_y_gradientes():
    torch.manual_seed(0)
    x = (torch.rand(3, 8) * torch.pi).requires_grad_()
    xq = x.detach().clone().requires_grad_()
    mio, ref = QNNTorch(8, semilla=1), crear_qnn(8, semilla=1)
    ym, yr = mio(x), ref(xq)
    assert ym.shape == (3, 1)
    torch.testing.assert_close(ym, yr, atol=1e-5, rtol=0)
    (ym**2).sum().backward()
    (yr**2).sum().backward()
    torch.testing.assert_close(mio.weight.grad, ref.weight.grad, atol=1e-4, rtol=0)
    torch.testing.assert_close(x.grad, xq.grad, atol=1e-4, rtol=0)
