import torch
import torch.nn as nn

from config.config import FCNNConfig
from config.dataset_config import general_dataset_configs


class CustomELMModel(nn.Module):
    def __init__(self):
        super(CustomELMModel, self).__init__()
        self.cfg = FCNNConfig().parse()
        gen_ds_cfg = general_dataset_configs(self.cfg)

        input_neurons = gen_ds_cfg.get("num_features")
        eq_neurons = gen_ds_cfg.get("eq_neurons")
        num_classes = gen_ds_cfg.get("num_classes")

        # Randomly initialize the weights and biases connecting input to first hidden layer
        self.fc1 = nn.Linear(input_neurons, eq_neurons[1], bias=False)
        self.fc2 = nn.Linear(eq_neurons[1], eq_neurons[2])
        self.fc3 = nn.Linear(eq_neurons[2], num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.LeakyReLU(negative_slope=self.cfg.slope)(x)
        x = self.fc2(x)
        x = nn.LeakyReLU(negative_slope=self.cfg.slope)(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x
