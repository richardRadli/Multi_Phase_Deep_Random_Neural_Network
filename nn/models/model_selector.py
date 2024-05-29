from abc import ABC, abstractmethod

from nn.models.vit import ViT
from nn.models.vit_elm import ViTELM
from utils.utils import use_gpu_if_available


class BaseModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass


class ViTModelWrapper(BaseModel):
    def __init__(self, vit_model_name, num_classes):
        self.model = ViT(vit_model_name, num_classes)

    def forward(self, x):
        return self.model(x)


class ViTELMModelWrapper(BaseModel):
    def __init__(self, vit_model_name, num_neurons, num_classes):
        self.model = ViTELM(vit_model_name, num_neurons, num_classes)

    def forward(self, x):
        return self.model(x)


class ModelFactory:
    @staticmethod
    def create_model(network_type, vit_model_name, num_neurons, num_classes, device=None):
        if network_type == "ViT":
            model = ViTModelWrapper(vit_model_name, num_classes).model
        elif network_type == "ViTELM":
            model = ViTELMModelWrapper(vit_model_name, num_neurons, num_classes).model
        else:
            raise ValueError(f"Network type {network_type} not supported")

        if device is None:
            device = use_gpu_if_available()

        model.to(device)
        return model
