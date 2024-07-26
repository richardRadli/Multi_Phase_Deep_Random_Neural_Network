from abc import ABC, abstractmethod

from nn.models.mpdrnn_model import (MultiPhaseDeepRandomizedNeuralNetworkBase,
                                    MultiPhaseDeepRandomizedNeuralNetworkSubsequent,
                                    MultiPhaseDeepRandomizedNeuralNetworkFinal)


class ModelSelector(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass


class BaseModelWrapper(ModelSelector):
    def __init__(self, network_cfg):
        self.model = MultiPhaseDeepRandomizedNeuralNetworkBase(
            num_data=network_cfg.get("first_layer_num_data"),
            num_features=network_cfg.get("first_layer_num_features"),
            hidden_nodes=network_cfg.get("first_layer_num_hidden"),
            output_nodes=network_cfg.get("first_layer_output_nodes"),
            activation_function=network_cfg.get("activation"),
            method=network_cfg.get("method"),
            rcond=network_cfg.get("rcond"),
            penalty_term=network_cfg.get("penalty_term"),
        )

    def forward(self, x):
        return self.model(x)


class SequentModelWrapper(ModelSelector):
    def __init__(self, network_cfg):
        self.model = MultiPhaseDeepRandomizedNeuralNetworkSubsequent(
            base_instance=network_cfg.get('initial_model'),
            mu=network_cfg.get('mu'),
            sigma=network_cfg.get('sigma')
        )

    def forward(self, x):
        return self.model(x)


class FinalModelWrapper(ModelSelector):
    def __init__(self, network_cfg):
        self.model = MultiPhaseDeepRandomizedNeuralNetworkFinal(
            subsequent_instance=network_cfg.get('subsequent_model'),
            mu=network_cfg.get("mu"),
            sigma=network_cfg.get("sigma"))

    def forward(self, x):
        return self.model(x)


class ModelFactory:
    @staticmethod
    def create(network_type, network_cfg):
        if network_type == "MultiPhaseDeepRandomizedNeuralNetworkBase":
            model = BaseModelWrapper(network_cfg).model
        elif network_type == "MultiPhaseDeepRandomizedNeuralNetworkSubsequent":
            model = SequentModelWrapper(network_cfg).model
        elif network_type == "MultiPhaseDeepRandomizedNeuralNetworkFinal":
            model = FinalModelWrapper(network_cfg).model
        else:
            raise ValueError(f"Network type {network_type} not supported")

        return model
