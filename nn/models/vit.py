import torch
import torch.nn as nn

from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32


class ViT(nn.Module):
    def __init__(self, vit_model_name, num_classes):
        super(ViT, self).__init__()
        self.model = self.build_model(vit_model_name, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: input tensor

        Returns:
            The output of the model.
        """

        return self.model(x)

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

    def build_model(self, vit_model_name, num_classes):
        """
        Modifies the selected ViT model's classification layer, by changing the number of classes.

        Returns:
            The modified ViT model.
        """

        model = self.select_vit_model(vit_model_name)
        layer = model.encoder.ln
        shape_layer = layer.weight.shape
        in_features = int(shape_layer[0])
        model.heads = nn.Linear(in_features=in_features, out_features=num_classes)

        return model
