import torch.nn as nn

from config.config import MPDRNNConfig


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
            "identity": nn.Identity(),
            "ReLU": nn.ReLU(),
            "leaky_ReLU": nn.LeakyReLU(negative_slope=self.cfg.slope),
            "tanh": nn.Tanh()
        }

        return activation_map[activation]
