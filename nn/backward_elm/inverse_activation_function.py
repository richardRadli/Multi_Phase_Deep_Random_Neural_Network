import torch


class InverseLeakyReLU(torch.nn.Module):
    def __init__(self, alpha):
        super(InverseLeakyReLU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x < 0, x * (1 / self.alpha), x)


class Logit(torch.nn.Module):
    def __init__(self):
        super(Logit, self).__init__()

    @staticmethod
    def forward(x):
        return torch.log(x / (1 - x))


class ArcTanh(torch.nn.Module):
    def __init__(self):
        super(ArcTanh, self).__init__()

    @staticmethod
    def forward(x):
        return torch.atanh(x)
