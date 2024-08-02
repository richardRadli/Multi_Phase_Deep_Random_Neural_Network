import torch
import torch.nn as nn


class FullyConnectedNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(FullyConnectedNeuralNetwork, self).__init__()
        """
        Initializes the Fully Connected Neural Network (FCNN) model.

        Args:
            input_size: The number of input features.
            hidden_size: The number of neurons in the hidden layers.
            output_size: The number of output features (classes).

        Returns:
            None
        """

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the network.
        """

        x = self.fc1(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.fc2(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.fc3(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.fc4(x)

        return x
