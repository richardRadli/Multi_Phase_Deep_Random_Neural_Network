from abc import ABC, abstractmethod

import torch

from nn.models.dev_drnn_model import (
    DevDeepRandomizedNeuralNetworkFirstLayer,
    DevDeepRandomizedNeuralNetworkSecondLayer,
    DevDeepRandomizedNeuralNetworkThirdLayer,
)


class LayerSelector(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass


class FirstLayerWrapper(LayerSelector):
    def __init__(self, network_cfg, train_loader):
        self.layer = DevDeepRandomizedNeuralNetworkFirstLayer(
            num_data=network_cfg.get("first_layer_num_data"),
            num_features=network_cfg.get("first_layer_num_features"),
            hidden_nodes=network_cfg.get("list_of_hidden_neurons"),
            output_nodes=network_cfg.get("first_layer_output_nodes"),
            activation_function=network_cfg.get("activation"),
            rcond=network_cfg.get("rcond"),
            penalty_term=network_cfg.get("penalty_term"),
            method=network_cfg.get("method"),
            train_loader=train_loader
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class SecondLayerWrapper(LayerSelector):
    def __init__(self, network_cfg):
        self.layer = DevDeepRandomizedNeuralNetworkSecondLayer(
            first_layer_instance=network_cfg.get("first_layer_instance"),
            mu=network_cfg.get("mu"),
            sigma=network_cfg.get("sigma"),
            scaling_factor=network_cfg.get("scaling_factor"),
            error_threshold=network_cfg.get("error_threshold"),
            similarity_threshold=network_cfg.get("similarity_threshold"),
            res_rcond=network_cfg.get("res_rcond")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class ThirdLayer(LayerSelector):
    def __init__(self, network_cfg: dict):
        self.layer = DevDeepRandomizedNeuralNetworkThirdLayer(
            second_layer_instance=network_cfg.get("second_layer_instance"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class LayerFactory:
    _layer_map = {
        "DevDeepRandomizedNeuralNetworkFirstLayer": FirstLayerWrapper,
        "DevDeepRandomizedNeuralNetworkSecondLayer": SecondLayerWrapper,
        "DevDeepRandomizedNeuralNetworkThirdLayer": ThirdLayer,
    }

    @staticmethod
    def create(network_type: str, network_cfg: dict, **kwargs):
        if network_type not in LayerFactory._layer_map:
            raise ValueError(f"Invalid layer type: {network_type}")

        layer_wrapper_class = LayerFactory._layer_map[network_type]
        layer = layer_wrapper_class(network_cfg, **kwargs).layer

        return layer
