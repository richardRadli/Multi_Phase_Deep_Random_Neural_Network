import torch
import torch.nn as nn

from torchvision.models import vit_b_16


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

        self.activation = nn.ReLU()

    def forward(self, x):
        hidden_layer = self.activation(x.matmul(self.alpha_weights))
        output = hidden_layer.matmul(self.beta_weights)
        return output

    def fit(self, x, y):
        hidden_layer = self.activation(x.matmul(self.alpha_weights))
        pseudo_inverse_hidden_layer = torch.pinverse(hidden_layer)
        self.beta_weights = nn.Parameter(pseudo_inverse_hidden_layer.matmul(y), requires_grad=False)


class ViTELM(nn.Module):
    def __init__(self, num_classes):
        super(ViTELM, self).__init__()

        self.vit_model = vit_b_16(weights="DEFAULT")
        self.vit_model.heads = nn.Identity()
        self.elm_head = ELM(num_input_neurons=self.get_input_neurons(),
                            num_hidden_neurons=self.get_input_neurons(),
                            num_classes=num_classes)

    def forward(self, x):
        features = self.vit_model(x)
        output = self.elm_head(features)
        return output

    def get_input_neurons(self):
        layer = self.vit_model.encoder.ln
        shape_layer = layer.weight.shape
        return int(shape_layer[0])

    def extract_vit_features(self, x):
        return self.vit_model(x)

    def train_elm(self, x, y):
        self.elm_head.fit(x, y)


