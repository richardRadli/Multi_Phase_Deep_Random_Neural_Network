import colorama
import os
import logging
import torch

from tqdm import tqdm
from typing import Any

from config.json_config import json_config_selector
from config.dataset_config import general_dataset_configs, drnn_paths_config
from nn.models.model_selector import ModelFactory
from utils.utils import (average_columns_in_excel, create_timestamp, insert_data_to_excel, setup_logger,
                         load_config_json, reorder_metrics_lists, create_train_valid_test_datasets, get_num_of_neurons)


class IPMPDRNN:
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- __I N I T__ ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        # Create timestamp as the program begins to execute.
        timestamp = create_timestamp()

        # Setup logger and colour
        setup_logger()
        colorama.init()

        # Initialize paths and settings
        self.cfg = (
            load_config_json(
                json_schema_filename=json_config_selector("ipmpdrnn").get("schema"),
                json_filename=json_config_selector("ipmpdrnn").get("config")
            )
        )

        # Boilerplate
        self.dataset_name = self.cfg.get("dataset_name")
        self.method = self.cfg.get('method')

        self.penalty_term = self.cfg.get('penalty')
        self.rcond = self.cfg.get("rcond").get(self.method)
        self.activation = self.cfg.get('activation')
        self.gen_ds_cfg = general_dataset_configs(self.cfg.get("dataset_name"))

        self.initial_model = None
        self.subsequent_model = None

        drnn_config = drnn_paths_config(self.dataset_name)
        sp = self.cfg.get('subset_percentage')

        # Save path
        self.save_filename = (
            os.path.join(
                drnn_config.get("ipmpdrnn").get("path_to_results"),
                f"{timestamp}_ipmpdrnn_{self.dataset_name}_{self.method}_sp_{sp}_rcond_{self.rcond}.xlsx")
        )

        # Set seed
        if self.cfg.get("seed"):
            torch.manual_seed(1234)

        # Load dataset
        file_path = general_dataset_configs(self.cfg.get('dataset_name')).get("cached_dataset_file")
        self.train_loader, self.valid_loader, self.test_loader = create_train_valid_test_datasets(file_path)

    def get_network_config(self, network_type: str, aux_net: bool = None) -> dict:
        """
        Retrieves the configuration for a specified network type.

        Args:
            network_type (str): The type of network for which to retrieve the configuration.
                                It should be one of the following:
                                - "MultiPhaseDeepRandomizedNeuralNetworkBase"
                                - "MultiPhaseDeepRandomizedNeuralNetworkSubsequent"
                                - "MultiPhaseDeepRandomizedNeuralNetworkFinal"
            aux_net (bool): Whether the network is an auxiliary network.

        Returns:
            Dict[str, Any]: A dictionary containing the configuration parameters for the specified network type.
                            The structure of the dictionary depends on the `network_type` value.
        """

        neurons = get_num_of_neurons(self.cfg, self.method)
        if aux_net:
            neurons = [h // self.cfg.get("num_aux_net") for h in neurons]

        net_cfg = {
            "MultiPhaseDeepRandomizedNeuralNetworkBase": {
                "first_layer_num_data": self.gen_ds_cfg.get("num_train_data"),
                "first_layer_num_features": self.gen_ds_cfg.get("num_features"),
                "list_of_hidden_neurons": neurons,
                "first_layer_output_nodes": self.gen_ds_cfg.get("num_classes"),
                "activation": self.activation,
                "method": self.method,
                "rcond": self.rcond,
                "penalty_term": self.penalty_term,
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

    def model_training_and_evaluation(self, model, weights, num_hidden_layers: int, verbose: bool):
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
                dataloader=self.test_loader,
                operation="test",
                layer_weights=weights,
                num_hidden_layers=num_hidden_layers,
                verbose=verbose
            )
        )

        return model, train_metrics, test_metrics

    def pruning_model(self, model: Any, weight_attr: str, set_weights_to_zero: bool):
        """
        Applies pruning to a model's weights based on the specified pruning method and subset percentage.

        Args:
            model: The model to be pruned. The model must have a `pruning` method and attributes corresponding
                   to weight matrices.
            weight_attr: The name of the attribute in the model that contains the weights to be pruned.
                         This should be a string representing the weight attribute's name.
            set_weights_to_zero: A boolean flag indicating whether to set the pruned weights to zero.

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
                pruning_percentage=self.cfg.get("subset_percentage")
            )
        )

        if set_weights_to_zero:
            weight_tensor = getattr(model, weight_attr).data
            weight_tensor[:, least_important_prune_indices] = 0
            return model, least_important_prune_indices
        else:
            return least_important_prune_indices

    def prune_initial_model(self, model: Any, set_weights_to_zero: bool):
        """
        Prunes the initial model's alpha weights based on the specified parameters.

        Args:
            model: The initial model to be pruned. It should have an attribute 'alpha_weights' and a
                   `pruning_model` method that can perform pruning.
            set_weights_to_zero: If True, pruned weights will be set to zero; otherwise, only the indices
                                 of pruned weights will be returned.

        Returns:
            A tuple containing:
                - The pruned model with weights set to zero (if `set_weights_to_zero` is True) or
                - A list of indices of the pruned weights (if `set_weights_to_zero` is False).
        """

        return self.pruning_model(model, weight_attr="alpha_weights", set_weights_to_zero=set_weights_to_zero)

    def prune_subsequent_model(self, model, set_weights_to_zero: bool):
        """
        Prunes the subsequent model's extended beta weights based on the specified parameters.

        Args:
            model: The subsequent model to be pruned. It should have an attribute 'extended_beta_weights' and a
                   `pruning_model` method that can perform pruning.
            set_weights_to_zero: If True, pruned weights will be set to zero; otherwise, only the indices
                                 of pruned weights will be returned.

        Returns:
            A tuple containing:
                - The pruned model with weights set to zero (if `set_weights_to_zero` is True) or
                - A list of indices of the pruned weights (if `set_weights_to_zero` is False).
        """

        return self.pruning_model(model, weight_attr="extended_beta_weights", set_weights_to_zero=set_weights_to_zero)

    def prune_final_model(self, model, set_weights_to_zero: bool):
        """
        Prunes the final model's extended gamma weights based on the specified parameters.

        Args:
            model: The subsequent model to be pruned. It should have an attribute 'extended_gamma_weights' and a
                   `pruning_model` method that can perform pruning.
            set_weights_to_zero: If True, pruned weights will be set to zero; otherwise, only the indices
                                 of pruned weights will be returned.

        Returns:
            A tuple containing:
                - The pruned model with weights set to zero (if `set_weights_to_zero` is True) or
                - A list of indices of the pruned weights (if `set_weights_to_zero` is False).
        """

        return self.pruning_model(model, weight_attr="extended_gamma_weights", set_weights_to_zero=set_weights_to_zero)

    def create_train_prune_aux_model(self, model: Any, model_type: str, weight_attr: str,
                                     least_important_prune_indices: list, num_aux_models: int):
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

        Returns:
            The updated original model with weights set based on the pruned auxiliary model.
        """

        net_cfg = self.get_network_config(model_type, aux_net=True)
        all_best_weights = []

        for _ in range(num_aux_models):
            aux_model = ModelFactory.create(model_type, net_cfg)
            aux_model.train_layer(self.train_loader)

            most_important_prune_indices, _ = (
                aux_model.pruning(
                    pruning_percentage=self.cfg.get("subset_percentage")
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

    def create_train_prune_initial_aux_model(self, model: Any, model_type: str, least_important_prune_indices: list):
        """
        Creates and trains an auxiliary model, prunes it, and updates the weights of the original model
        using the pruned weights from the auxiliary model. Specifically handles the 'alpha_weights' attribute
        of the model.

        Args:
            model: The original model to be updated. It should have an attribute named 'alpha_weights'
                   and a `pruning` method.
            model_type: A string indicating the type of the auxiliary model to be created.
            least_important_prune_indices: A list of indices of the least important weights to be pruned.

        Returns:
            The updated original model with weights set based on the pruned auxiliary model.
        """

        return self.create_train_prune_aux_model(model,
                                                 model_type,
                                                 weight_attr="alpha_weights",
                                                 least_important_prune_indices=least_important_prune_indices,
                                                 num_aux_models=2)

    def create_train_prune_subsequent_aux_model(self, model: Any, model_type: str, least_important_prune_indices: list):
        """
        Creates and trains an auxiliary model, prunes it, and updates the weights of the original model
        using the pruned weights from the auxiliary model. Specifically handles the 'extended_beta_weights' attribute
        of the model.

        Args:
            model: The original model to be updated. It should have an attribute named 'extended_beta_weights'
                   and a `pruning` method.
            model_type: A string indicating the type of the auxiliary model to be created.
            least_important_prune_indices: A list of indices of the least important weights to be pruned.

        Returns:
            The updated original model with weights set based on the pruned auxiliary model.
        """

        return self.create_train_prune_aux_model(model,
                                                 model_type,
                                                 weight_attr="extended_beta_weights",
                                                 least_important_prune_indices=least_important_prune_indices,
                                                 num_aux_models=2)

    def create_train_prune_final_aux_model(self, model: Any, model_type: str, least_important_prune_indices: list):
        """
        Creates and trains an auxiliary model, prunes it, and updates the weights of the original model
        using the pruned weights from the auxiliary model. Specifically handles the 'extended_gamma_weights' attribute
        of the model.

        Args:
            model: The original model to be updated. It should have an attribute named 'extended_gamma_weights'
                   and a `pruning` method.
            model_type: A string indicating the type of the auxiliary model to be created.
            least_important_prune_indices: A list of indices of the least important weights to be pruned.

        Returns:
            The updated original model with weights set based on the pruned auxiliary model.
        """

        return self.create_train_prune_aux_model(model,
                                                 model_type,
                                                 weight_attr="extended_gamma_weights",
                                                 least_important_prune_indices=least_important_prune_indices,
                                                 num_aux_models=2)

    def main(self):
        """

        Returns:

        """

        # Load data
        training_time = []

        for i in tqdm(range(self.cfg.get('number_of_tests')), desc=colorama.Fore.CYAN + "Process"):
            # Create model

            net_cfg = self.get_network_config("MultiPhaseDeepRandomizedNeuralNetworkBase")
            self.initial_model = ModelFactory.create("MultiPhaseDeepRandomizedNeuralNetworkBase", net_cfg)

            # Train and evaluate the model
            self.initial_model, initial_model_training_metrics, initial_model_testing_metrics = (
                self.model_training_and_evaluation(
                    model=self.initial_model,
                    weights=self.initial_model.beta_weights,
                    num_hidden_layers=1,
                    verbose=True
                )
            )
            training_time.append(self.initial_model.train_ith_layer.execution_time)

            # Pruning
            least_important_prune_indices = (
                self.prune_initial_model(self.initial_model, set_weights_to_zero=False)
            )

            # Create aux model 1
            self.initial_model = (
                self.create_train_prune_initial_aux_model(model=self.initial_model,
                                                          model_type="MultiPhaseDeepRandomizedNeuralNetworkBase",
                                                          least_important_prune_indices=least_important_prune_indices)
            )
            training_time.append(self.initial_model.train_ith_layer.execution_time)

            (self.initial_model,
             initial_model_subs_weights_training_metrics,
             initial_model_subs_weights_testing_metrics) = (
                self.model_training_and_evaluation(
                    model=self.initial_model,
                    weights=self.initial_model.beta_weights,
                    num_hidden_layers=1,
                    verbose=True
                )
            )
            training_time.append(self.initial_model.train_ith_layer.execution_time)

            # Subsequent Model

            net_cfg = self.get_network_config("MultiPhaseDeepRandomizedNeuralNetworkSubsequent")
            self.subsequent_model = ModelFactory.create("MultiPhaseDeepRandomizedNeuralNetworkSubsequent", net_cfg)

            (self.subsequent_model,
             subsequent_model_subs_weights_training_metrics,
             subsequent_model_subs_weights_testing_metrics) = (
                self.model_training_and_evaluation(
                    model=self.subsequent_model,
                    weights=[self.subsequent_model.extended_beta_weights,
                             self.subsequent_model.gamma_weights],
                    num_hidden_layers=2,
                    verbose=True
                )
            )
            training_time.append(self.subsequent_model.train_ith_layer.execution_time)

            least_important_prune_indices = self.prune_subsequent_model(
                model=self.subsequent_model,
                set_weights_to_zero=False
            )

            self.subsequent_model = (
                self.create_train_prune_subsequent_aux_model(
                    model=self.subsequent_model,
                    model_type="MultiPhaseDeepRandomizedNeuralNetworkSubsequent",
                    least_important_prune_indices=least_important_prune_indices
                )
            )
            training_time.append(self.subsequent_model.train_ith_layer.execution_time)

            (self.subsequent_model,
             subsequent_model_model_subs_weights_training_metrics,
             subsequent_model_model_subs_weights_testing_metrics) = (
                self.model_training_and_evaluation(
                    model=self.subsequent_model,
                    weights=[self.subsequent_model.extended_beta_weights,
                             self.subsequent_model.gamma_weights],
                    num_hidden_layers=2,
                    verbose=True
                )
            )
            training_time.append(self.subsequent_model.train_ith_layer.execution_time)

            # Final Model

            net_cfg = self.get_network_config(network_type="MultiPhaseDeepRandomizedNeuralNetworkFinal")
            final_model = ModelFactory.create("MultiPhaseDeepRandomizedNeuralNetworkFinal", net_cfg)

            (final_model,
             final_model_subs_weights_training_metrics,
             final_model_subs_weights_testing_metrics) = (
                self.model_training_and_evaluation(
                    model=final_model,
                    weights=[final_model.extended_beta_weights,
                             final_model.extended_gamma_weights,
                             final_model.delta_weights],
                    num_hidden_layers=3,
                    verbose=True
                )
            )
            training_time.append(final_model.train_ith_layer.execution_time)

            least_important_prune_indices = self.prune_final_model(
                model=final_model,
                set_weights_to_zero=False
            )

            final_model = (
                self.create_train_prune_final_aux_model(
                    model=final_model,
                    model_type="MultiPhaseDeepRandomizedNeuralNetworkFinal",
                    least_important_prune_indices=least_important_prune_indices
                )
            )
            training_time.append(final_model.train_ith_layer.execution_time)

            (final_model,
             final_model_model_subs_weights_training_metrics,
             final_model_model_subs_weights_testing_metrics) = (
                self.model_training_and_evaluation(
                    model=final_model,
                    weights=[final_model.extended_beta_weights,
                             final_model.extended_gamma_weights,
                             final_model.delta_weights],
                    num_hidden_layers=3,
                    verbose=True
                )
            )
            training_time.append(final_model.train_ith_layer.execution_time)

            # Excel
            best = (
                final_model_subs_weights_testing_metrics) if (final_model_subs_weights_testing_metrics[0] >
                                                              final_model_model_subs_weights_testing_metrics[0]) else (
                final_model_model_subs_weights_testing_metrics
            )

            metrics = reorder_metrics_lists(train_metrics=final_model_model_subs_weights_training_metrics,
                                            test_metrics=best,
                                            training_time_list=training_time)
            insert_data_to_excel(self.save_filename, self.cfg.get("dataset_name"), i + 2, metrics)

            training_time.clear()

        average_columns_in_excel(self.save_filename)


if __name__ == "__main__":
    try:
        ipmpdrnn = IPMPDRNN()
        ipmpdrnn.main()
    except KeyboardInterrupt as kie:
        logging.error(f"Keyboard interrupt received: {kie}")
