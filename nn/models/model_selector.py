from abc import ABC, abstractmethod

import torch

from nn.models.mpdrnn_model import (MultiPhaseDeepRandomizedNeuralNetworkBase,
                                    MultiPhaseDeepRandomizedNeuralNetworkSubsequent,
                                    MultiPhaseDeepRandomizedNeuralNetworkFinal)


class ModelSelector(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass


class BaseModelWrapper(ModelSelector):
    def __init__(self, network_cfg):
        self.model = MultiPhaseDeepRandomizedNeuralNetworkBase(
            num_data=network_cfg.get("first_layer_num_data"),
            num_features=network_cfg.get("first_layer_num_features"),
            hidden_nodes=network_cfg.get("list_of_hidden_neurons"),
            output_nodes=network_cfg.get("first_layer_output_nodes"),
            activation_function=network_cfg.get("activation"),
            method=network_cfg.get("method"),
            rcond=network_cfg.get("rcond"),
            penalty_term=network_cfg.get("penalty_term"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SequentModelWrapper(ModelSelector):
    def __init__(self, network_cfg):
        self.model = MultiPhaseDeepRandomizedNeuralNetworkSubsequent(
            base_instance=network_cfg.get('initial_model'),
            mu=network_cfg.get('mu'),
            sigma=network_cfg.get('sigma')
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FinalModelWrapper(ModelSelector):
    def __init__(self, network_cfg: dict):
        self.model = MultiPhaseDeepRandomizedNeuralNetworkFinal(
            subsequent_instance=network_cfg.get('subsequent_model'),
            mu=network_cfg.get("mu"),
            sigma=network_cfg.get("sigma")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ModelFactory:
    _model_map = {
        "MultiPhaseDeepRandomizedNeuralNetworkBase": BaseModelWrapper,
        "MultiPhaseDeepRandomizedNeuralNetworkSubsequent": SequentModelWrapper,
        "MultiPhaseDeepRandomizedNeuralNetworkFinal": FinalModelWrapper,
    }

    @staticmethod
    def create(network_type: str, network_cfg: dict):
        if network_type not in ModelFactory._model_map:
            raise ValueError(f"Invalid model type: {network_type}")

        model_wrapper_class = ModelFactory._model_map[network_type]

        model = model_wrapper_class(network_cfg).model

        return model