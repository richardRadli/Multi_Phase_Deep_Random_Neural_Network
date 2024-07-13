import torch
import torch.nn as nn


class FCNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.fc2(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.fc3(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)

        return x
