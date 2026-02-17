import torch

from config.json_config import json_config_selector
from config.dataset_config import general_dataset_configs
from utils.utils import create_train_valid_test_datasets, load_config_json, setup_logger


class BaseDevDRNN:
    def __init__(self):
        setup_logger()

        # Initialize paths and settings
        self.cfg = (
            load_config_json(
                json_schema_filename=json_config_selector("dev_drnn").get("schema"),
                json_filename=json_config_selector("dev_drnn").get("config"),
            )
        )

        if self.cfg.get("seed"):
            torch.manual_seed(1234)

        dataset_name = self.cfg.get("dataset_name")
        self.activation = self.cfg.get('activation')
        self.gen_ds_cfg = general_dataset_configs(self.cfg.get("dataset_name"))

        self.first_layer = None
        self.second_layer = None

        file_path = general_dataset_configs(dataset_name).get("cached_dataset_file")
        self.train_loader, self.valid_loader, self.test_loader = create_train_valid_test_datasets(file_path)

    def get_network_config(self, network_type, config):
        """
        Retrieves the configuration for a specified network type.

        Args:
            network_type (str): The type of network for which to retrieve the configuration.
                                It should be one of the following:
                                - "DevDeepRandomizedNeuralNetworkFirstLayer"
                                - "DevDeepRandomizedNeuralNetworkSecondLayer"
                                - "DevDeepRandomizedNeuralNetworkThirdLayer"
            config (dict): A dictionary containing the configuration of the network.

        Returns:
            Dict[str, Any]: A dictionary containing the configuration parameters for the specified network type.
                            The structure of the dictionary depends on the `network_type` value.
        """

        net_cfg = {
            "DevDeepRandomizedNeuralNetworkFirstLayer": {
                "first_layer_num_data": self.gen_ds_cfg.get("num_train_data"),
                "first_layer_num_features": self.gen_ds_cfg.get("num_features"),
                "list_of_hidden_neurons": config.get("neurons"),
                "first_layer_output_nodes": self.gen_ds_cfg.get("num_classes"),
                "activation": self.activation,
                "rcond": config.get("rcond"),
                "method": config.get("method"),
                "penalty_term": config.get("penalty_term")
            },
            "DevDeepRandomizedNeuralNetworkSecondLayer": {
                "first_layer_instance": self.first_layer,
                "sigma": self.cfg.get('sigma'),
                "mu": self.cfg.get('mu')
            },
            "DevDeepRandomizedNeuralNetworkThirdLayer": {
                "second_layer_instance": self.second_layer,
            }
        }

        return net_cfg[network_type]

    def model_training_and_evaluation(
        self,
        model,
        weights,
        num_hidden_layers,
        verbose
    ):
        """
        Trains and evaluates a given model on the training and testing datasets.

        Args:
            model: The model to be trained and evaluated. It should have methods
                   `train_layer` and `predict_and_evaluate`.
            weights: The weights to be used in the model's evaluation. This may
                     be used to initialize or modify the model.
            num_hidden_layers: The number of hidden layers in the model. This
                               parameter may affect the evaluation process.
            verbose: A boolean flag indicating whether to print detailed logs.

        Returns:
            A tuple containing:
                - The trained model.
                - A dictionary of training metrics.
                - A dictionary of testing metrics.
        """

        model.train_layer()

        training_metrics = model.predict_and_evaluate(
            dataloader=self.train_loader,
            operation="train",
            layer_weights=weights,
            num_hidden_layers=num_hidden_layers,
            verbose=verbose
        )

        testing_metrics = model.predict_and_evaluate(
            dataloader=self.test_loader,
            operation="test",
            layer_weights=weights,
            num_hidden_layers=num_hidden_layers,
            verbose=verbose
        )

        return model, training_metrics, testing_metrics
    