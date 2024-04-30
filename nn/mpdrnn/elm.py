import torch
import torch.nn as nn


class ELM(nn.Module):
    def __init__(self, activation_function):
        super(ELM, self).__init__()
        self.activation_function = self.get_activation(activation_function)

    def forward(self, x, weights):
        return self.activation_function(x @ weights)

    @staticmethod
    def get_activation(activation):
        activation_map = {
            "sigmoid": nn.Sigmoid(),
            "identity": nn.Identity(),
            "ReLU": nn.ReLU(),
            "leaky_ReLU": nn.LeakyReLU(),
            "tanh": nn.Tanh()
        }

        return activation_map[activation]
