import torch
from types import SimpleNamespace

class Dataset:
    class Treinamento:

        vetores = torch.frombuffer(open('train-images.idx3-ubyte','rb').read()[16:], dtype=torch.uint8).reshape((60_000, 784)) * 0.01 - 0.01
        gabarito = torch.zeros((60_000, 10))
        for i, j in enumerate(tuple(open('train-labels.idx1-ubyte', 'rb').read()[8:])):
            gabarito[i, j] = 1

class Nn:

    def __init__(self):

        self.pesos = SimpleNamespace(
            ocultos=torch.randn((784, 16)) * 0.05,
            saida = torch.randn((16, 10)) * 0.35
        )
        self.vieses = SimpleNamespace(
            ocultos=torch.randn(16),
            saida = torch.randn(10)
        )

class Propagation(Nn):

    def __init__(self, entrada: torch.Tensor):

        self.ativacoes = torch.relu(torch.layer_norm(entrada @ self.pesos.ocultos + self.vieses.ocultos))
        self.saida_simples = self.ativacoes @ self.pesos.saida + self.vieses.saida
        self.gradientes = SimpleNamespace(
            pesos=SimpleNamespace(ocultos=torch.zeros((784, 16)),
                                  saida=torch.zeros((16, 10))),
            vieses=SimpleNamespace(ocultos=torch.zeros(16),
                                   saida=torch.zeros(10))
        )

class Otimizador(Propagation):

    def __init__(self, entrada: torch.Tensor, otmzd: str):

        if otmzd not in ('nesterov', 'adam', 'sgd'):
            raise ValueError(f'Otimizador "{otmzd}" invalido')

        self.momento = SimpleNamespace(
            pesos=SimpleNamespace(ocultos=torch.zeros((784, 16)),
                                  saida=torch.zeros((16, 10))),
            vieses=SimpleNamespace(ocultos=torch.zeros(16),
                                   saida=torch.zeros(10))
        )if otmzd in ('nesterov', 'adam') else None

        self.tx_apd_adt = SimpleNamespace(
            pesos=SimpleNamespace(ocultos=torch.zeros((784, 16)),
                                  saida=torch.zeros((16, 10))),
            vieses=SimpleNamespace(ocultos=torch.zeros(16),
                                   saida=torch.zeros(10))
        )if otmzd == 'adam' else None