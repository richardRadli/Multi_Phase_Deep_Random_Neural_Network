import torch.nn as nn

from config.config import MPDRNNConfig
from nn.backward_elm.inverse_activation_function import ArcTanh, InverseLeakyReLU, Logit


class ELM(nn.Module):
    def __init__(self, activation_function):
        super(ELM, self).__init__()
        self.cfg = MPDRNNConfig().parse()

        self.activation_function = self.get_activation(activation_function)

    def forward(self, x, weights):
        return self.activation_function(x @ weights)

    def get_activation(self, activation):
        activation_map = {
            "sigmoid": nn.Sigmoid(),
            "logit": Logit(),
            "identity": nn.Identity(),
            "ReLU": nn.ReLU(),
            "leaky_ReLU": nn.LeakyReLU(negative_slope=self.cfg.slope),
            "inverse_leaky_ReLU": InverseLeakyReLU(alpha=self.cfg.slope),
            "tanh": nn.Tanh(),
            "atanh": ArcTanh()
        }

        return activation_map[activation]
