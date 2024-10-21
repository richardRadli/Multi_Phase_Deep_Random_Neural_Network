import torch

from typing import Any

from config.json_config import json_config_selector
from config.dataset_config import general_dataset_configs
from nn.models.model_selector import ModelFactory
from utils.utils import (setup_logger, load_config_json, create_train_valid_test_datasets, measure_execution_time)


class BaseIPMPDRNN:
    def __init__(self):
        # Setup logger and colour
        setup_logger()

        # Initialize paths and settings
        self.cfg = (
            load_config_json(
                json_schema_filename=json_config_selector("ipmpdrnn").get("schema"),
                json_filename=json_config_selector("ipmpdrnn").get("config")
            )
        )

        self.dataset_name = self.cfg.get("dataset_name")
        self.method = self.cfg.get('method')
        self.activation = self.cfg.get('activation')
        self.gen_ds_cfg = general_dataset_configs(self.cfg.get("dataset_name"))

        self.initial_model = None
        self.subsequent_model = None

        # Set seed
        if self.cfg.get("seed"):
            torch.manual_seed(1234)

        file_path = general_dataset_configs(self.cfg.get('dataset_name')).get("cached_dataset_file")
        self.train_loader, self.valid_loader, self.test_loader = create_train_valid_test_datasets(file_path)

    def get_network_config(self, network_type: str, config, aux_net: bool = None) -> dict:
        neurons = config.get("neurons")
        if aux_net:
            neurons = [h // config.get("num_aux_net") for h in neurons]

        net_cfg = {
            "MultiPhaseDeepRandomizedNeuralNetworkBase": {
                "first_layer_num_data": self.gen_ds_cfg.get("num_train_data"),
                "first_layer_num_features": self.gen_ds_cfg.get("num_features"),
                "list_of_hidden_neurons": neurons,
                "first_layer_output_nodes": self.gen_ds_cfg.get("num_classes"),
                "activation": self.activation,
                "method": self.method,
                "rcond": config.get("rcond"),
                "penalty_term": config.get("penalty_term"),
            },
            "MultiPhaseDeepRandomizedNeuralNetworkSubsequent": {
                "initial_model": self.initial_model,
                "sigma": self.cfg.get('sigma'),
                "mu": self.cfg.get('mu')
            },
            "MultiPhaseDeepRandomizedNeuralNetworkFinal": {
                "subsequent_model": self.subsequent_model,
                "sigma": self.cfg.get('sigma'),
                "mu": self.cfg.get('mu')
            }
        }

        return net_cfg[network_type]

    def model_training_and_evaluation(self, model, eval_set, weights, num_hidden_layers: int, verbose: bool):
        """
        Trains and evaluates a given model on the training and testing datasets.

        Args:
            model: The model to be trained and evaluated. It should have methods
                   `train_layer` and `predict_and_evaluate`.
            eval_set: Evaluation dataset, either validation or test set.
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

        model.train_layer(self.train_loader)

        train_metrics = (
            model.predict_and_evaluate(
                dataloader=self.train_loader,
                operation="train",
                layer_weights=weights,
                num_hidden_layers=num_hidden_layers,
                verbose=verbose
            )
        )
        test_metrics = (
            model.predict_and_evaluate(
                dataloader=eval_set,
                operation="test",
                layer_weights=weights,
                num_hidden_layers=num_hidden_layers,
                verbose=verbose
            )
        )

        return model, train_metrics, test_metrics

    @staticmethod
    def pruning_model(model: Any, weight_attr: str, set_weights_to_zero: bool, config):
        """
        Applies pruning to a model's weights based on the specified pruning method and subset percentage.

        Args:
            model: The model to be pruned. The model must have a `pruning` method and attributes corresponding
                   to weight matrices.
            weight_attr: The name of the attribute in the model that contains the weights to be pruned.
                         This should be a string representing the weight attribute's name.
            set_weights_to_zero: A boolean flag indicating whether to set the pruned weights to zero.
            config: The configuration of the pruning method.

        Returns:
            If `set_weights_to_zero` is True:
                A tuple containing:
                    - The updated model with pruned weights set to zero.
                    - A list of indices of the pruned weights.
            If `set_weights_to_zero` is False:
                A list of indices of the pruned weights.
        """

        _, least_important_prune_indices = (
            model.pruning(
                pruning_percentage=config.get("subset_percentage")
            )
        )

        if set_weights_to_zero:
            weight_tensor = getattr(model, weight_attr).data
            weight_tensor[:, least_important_prune_indices] = 0
            return model, least_important_prune_indices
        else:
            return least_important_prune_indices

    @measure_execution_time
    def prune_initial_model(self, model: Any, set_weights_to_zero: bool, config):
        """
        Prunes the initial model's alpha weights based on the specified parameters.

        Args:
            model: The initial model to be pruned. It should have an attribute 'alpha_weights' and a
                   `pruning_model` method that can perform pruning.
            set_weights_to_zero: If True, pruned weights will be set to zero; otherwise, only the indices
                                 of pruned weights will be returned.
            config: The configuration of the pruning method.

        Returns:
            A tuple containing:
                - The pruned model with weights set to zero (if `set_weights_to_zero` is True) or
                - A list of indices of the pruned weights (if `set_weights_to_zero` is False).
        """

        return self.pruning_model(
            model, weight_attr="alpha_weights", set_weights_to_zero=set_weights_to_zero, config=config
        )

    @measure_execution_time
    def prune_subsequent_model(self, model, set_weights_to_zero: bool, config):
        """
        Prunes the subsequent model's extended beta weights based on the specified parameters.

        Args:
            model: The subsequent model to be pruned. It should have an attribute 'extended_beta_weights' and a
                   `pruning_model` method that can perform pruning.
            set_weights_to_zero: If True, pruned weights will be set to zero; otherwise, only the indices
                                 of pruned weights will be returned.
            config: The configuration of the pruning method.

        Returns:
            A tuple containing:
                - The pruned model with weights set to zero (if `set_weights_to_zero` is True) or
                - A list of indices of the pruned weights (if `set_weights_to_zero` is False).
        """

        return self.pruning_model(
            model, weight_attr="extended_beta_weights", set_weights_to_zero=set_weights_to_zero, config=config
        )

    @measure_execution_time
    def prune_final_model(self, model, set_weights_to_zero: bool, config):
        """
        Prunes the final model's extended gamma weights based on the specified parameters.

        Args:
            model: The subsequent model to be pruned. It should have an attribute 'extended_gamma_weights' and a
                   `pruning_model` method that can perform pruning.
            set_weights_to_zero: If True, pruned weights will be set to zero; otherwise, only the indices
                                 of pruned weights will be returned.
            config:

        Returns:
            A tuple containing:
                - The pruned model with weights set to zero (if `set_weights_to_zero` is True) or
                - A list of indices of the pruned weights (if `set_weights_to_zero` is False).
        """

        return self.pruning_model(
            model, weight_attr="extended_gamma_weights", set_weights_to_zero=set_weights_to_zero, config=config
        )

    def create_train_prune_aux_model(self, model: Any, model_type: str, weight_attr: str,
                                     least_important_prune_indices: list, config):
        """
        Creates and trains an auxiliary model, prunes it, and uses the weights from the pruned auxiliary model
        to update the weights of the original model.

        Args:
            model: The original model to be updated. It should have an attribute specified by `weight_attr`
                   and a `pruning` method.
            model_type: A string indicating the type of the auxiliary model to be created.
            weight_attr: The attribute name of the model that holds the weights to be updated.
            least_important_prune_indices: Optional list of indices of the least important weights to be pruned.
                                            If not provided, the indices are computed by pruning the original model.
            config: The number of auxiliary models to be created.

        Returns:
            The updated original model with weights set based on the pruned auxiliary model.
        """

        net_cfg = self.get_network_config(model_type, config, aux_net=True)
        all_best_weights = []

        for _ in range(config.get("num_aux_net")):
            aux_model = ModelFactory.create(model_type, net_cfg)
            aux_model.train_layer(self.train_loader)

            most_important_prune_indices, _ = (
                aux_model.pruning(
                    pruning_percentage=config.get("subset_percentage")
                )
            )

            best_weight_tensor = getattr(aux_model, weight_attr).data
            best_weights = best_weight_tensor[:, most_important_prune_indices]
            all_best_weights.append(best_weights)

        best_weights_final = torch.cat(all_best_weights, dim=1)

        current_dim = best_weights_final.shape[1]
        required_dim = len(least_important_prune_indices)

        if current_dim > required_dim:
            best_weights_final = best_weights_final[:, :required_dim]
        elif current_dim < required_dim:
            padding = required_dim - current_dim
            padding_tensor = torch.zeros(best_weights_final.shape[0], padding)
            best_weights_final = torch.cat((best_weights_final, padding_tensor), dim=1)

        weight_tensor = getattr(model, weight_attr).data
        weight_tensor[:, least_important_prune_indices] = best_weights_final

        return model

    @measure_execution_time
    def create_train_prune_initial_aux_model(self, model: Any, model_type: str, least_important_prune_indices: list,
                                             config):
        """
        Creates and trains an auxiliary model, prunes it, and updates the weights of the original model
        using the pruned weights from the auxiliary model. Specifically handles the 'alpha_weights' attribute
        of the model.

        Args:
            model: The original model to be updated. It should have an attribute named 'alpha_weights'
                   and a `pruning` method.
            model_type: A string indicating the type of the auxiliary model to be created.
            least_important_prune_indices: A list of indices of the least important weights to be pruned.
            config: The number of auxiliary models to be created.
        Returns:
            The updated original model with weights set based on the pruned auxiliary model.
        """

        return self.create_train_prune_aux_model(model,
                                                 model_type,
                                                 weight_attr="alpha_weights",
                                                 least_important_prune_indices=least_important_prune_indices,
                                                 config=config)

    @measure_execution_time
    def create_train_prune_subsequent_aux_model(
            self, model: Any, model_type: str, least_important_prune_indices: list, config
    ):
        """
        Creates and trains an auxiliary model, prunes it, and updates the weights of the original model
        using the pruned weights from the auxiliary model. Specifically handles the 'extended_beta_weights' attribute
        of the model.

        Args:
            model: The original model to be updated. It should have an attribute named 'extended_beta_weights'
                   and a `pruning` method.
            model_type: A string indicating the type of the auxiliary model to be created.
            least_important_prune_indices: A list of indices of the least important weights to be pruned.
            config: The number of auxiliary models to be created.

        Returns:
            The updated original model with weights set based on the pruned auxiliary model.
        """
        return self.create_train_prune_aux_model(model,
                                                 model_type,
                                                 weight_attr="extended_beta_weights",
                                                 least_important_prune_indices=least_important_prune_indices,
                                                 config=config)

    @measure_execution_time
    def create_train_prune_final_aux_model(self, model: Any, model_type: str, least_important_prune_indices: list,
                                           config):
        """
        Creates and trains an auxiliary model, prunes it, and updates the weights of the original model
        using the pruned weights from the auxiliary model. Specifically handles the 'extended_gamma_weights' attribute
        of the model.

        Args:
            model: The original model to be updated. It should have an attribute named 'extended_gamma_weights'
                   and a `pruning` method.
            model_type: A string indicating the type of the auxiliary model to be created.
            least_important_prune_indices: A list of indices of the least important weights to be pruned.
            config: The number of auxiliary models to be created.

        Returns:
            The updated original model with weights set based on the pruned auxiliary model.
        """

        return self.create_train_prune_aux_model(model,
                                                 model_type,
                                                 weight_attr="extended_gamma_weights",
                                                 least_important_prune_indices=least_important_prune_indices,
                                                 config=config)
