import torch
import torch.nn as nn

from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32


class ELM(nn.Module):
    def __init__(self, num_input_neurons, num_hidden_neurons, num_classes):
        super(ELM, self).__init__()

        self.num_input_neurons = num_input_neurons
        self.num_hidden_neurons = num_hidden_neurons
        self.num_classes = num_classes

        self.alpha_weights = (
            nn.Parameter(torch.randn(self.num_input_neurons, self.num_hidden_neurons),
                         requires_grad=False)
        )
        self.beta_weights = (
            nn.Parameter(torch.randn(self.num_hidden_neurons, self.num_classes),
                         requires_grad=False)
        )

        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        hidden_layer = self.activation(x.matmul(self.alpha_weights))
        output = hidden_layer.matmul(self.beta_weights)
        return output

    def fit(self, x, y):
        hidden_layer = self.activation(x.matmul(self.alpha_weights))
        pseudo_inverse_hidden_layer = torch.pinverse(hidden_layer)
        self.beta_weights.data = pseudo_inverse_hidden_layer.matmul(y)


class ViTELM(nn.Module):
    def __init__(self, vit_model_name, num_classes):
        super(ViTELM, self).__init__()

        self.vit_model = self.select_vit_model(vit_model_name)
        self.vit_model.heads = nn.Sequential(
            nn.Identity(),
            ELM(num_input_neurons=self.get_input_neurons(),
                num_hidden_neurons=self.get_input_neurons(),
                num_classes=num_classes)
        )

    def forward(self, x):
        return self.vit_model(x)

    @staticmethod
    def select_vit_model(vit_model: str):
        """

        Args:
            vit_model: The name of the ViT model to use.

        Returns:
            The selected ViT model.
        """

        vit_model_dict = {
            "vitb16":
                vit_b_16(weights="DEFAULT"),
            "vitb32":
                vit_b_32(weights="DEFAULT"),
            "vitl16":
                vit_l_16(weights="DEFAULT"),
            "vitl32":
                vit_l_32(weights="DEFAULT"),
        }
        return vit_model_dict[vit_model]

    def get_input_neurons(self):
        layer = self.vit_model.encoder.ln
        shape_layer = layer.weight.shape
        return int(shape_layer[0])

    def extract_vit_features(self, x):
        return self.vit_model(x)

    def train_elm(self, x, y):
        self.vit_model.heads[1].fit(x, y)
