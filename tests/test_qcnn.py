import torch

from qcnn_ecg.modelos.qcnn import crear_qnn


def test_qcnn_forward_backward_y_paso_de_optimizacion():
    qnn = crear_qnn(8)
    x = (torch.rand(4, 8) * torch.pi).requires_grad_()
    y = qnn(x)
    assert y.shape == (4, 1) and y.abs().max() <= 1 + 1e-6

    y.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0  # gradiente hacia la entrada
    assert qnn.weight.grad is not None and qnn.weight.grad.abs().sum() > 0
    assert qnn.weight.numel() == 18

    antes = qnn.weight.detach().clone()
    torch.optim.Adam(qnn.parameters(), lr=0.1).step()
    assert not torch.equal(antes, qnn.weight.detach())
