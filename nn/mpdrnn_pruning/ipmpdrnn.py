import colorama
import os
import logging
import torch

from nn.dataloaders.npz_dataloader import NpzDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import general_dataset_configs, drnn_paths_config
from nn.models.model_selector import ModelFactory
from utils.utils import (average_columns_in_excel, create_timestamp, insert_data_to_excel, setup_logger,
                         load_config_json, reorder_metrics_lists)


class Experiment:
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
            load_config_json(json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_ipmdrnn"),
                             json_filename=JSON_FILES_PATHS.get_data_path("config_ipmdrnn"))
        )

        # Boilerplate
        self.dataset_name = self.cfg.get("dataset_name")
        self.method = self.cfg.get('method')
        self.penalty_term = self.cfg.get('penalty').get(self.dataset_name)
        self.rcond = self.cfg.get("rcond").get(self.dataset_name).get(self.cfg.get("method"))
        self.activation = self.cfg.get('activation')
        self.gen_ds_cfg = general_dataset_configs(self.cfg.get("dataset_name"))

        self.initial_model = None
        self.subsequent_model = None

        drnn_config = drnn_paths_config(self.cfg.get("dataset_name"))
        sp = self.cfg.get('subset_percentage').get(self.dataset_name)
        pm = self.cfg.get('pruning_method')

        # Save path
        self.save_filename = (
            os.path.join(
                drnn_config.get("ipmpdrnn").get("path_to_results"),
                f"{timestamp}_ipmpdrnn_{self.dataset_name}_{self.method}_sp_{sp}_pm_{pm}_rcond_{self.rcond}.xlsx")
        )

        # Set seed
        if self.cfg.get("seed"):
            torch.manual_seed(1234)

        # Load dataset
        file_path = general_dataset_configs(self.cfg.get('dataset_name')).get("cached_dataset_file")
        self.train_loader, self.valid_loader, self.test_loader = (
            self.create_train_valid_test_datasets(file_path)
        )

    def get_num_of_neurons(self, method, dataset_name):
        num_neurons = {
            "BASE": self.cfg.get("eq_neurons").get(dataset_name),
            "EXP_ORT": self.cfg.get("exp_neurons").get(dataset_name),
            "EXP_ORT_C": self.cfg.get("exp_neurons").get(dataset_name),
        }
        return num_neurons[method]

    def get_network_config(self, network_type):
        net_cfg = {
            "MultiPhaseDeepRandomizedNeuralNetworkBase": {
                "first_layer_num_data": self.gen_ds_cfg.get("num_train_data"),
                "first_layer_num_features": self.gen_ds_cfg.get("num_features"),
                "first_layer_num_hidden": self.get_num_of_neurons(self.cfg.get("method"), self.dataset_name),
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

    @staticmethod
    def create_train_valid_test_datasets(file_path):
        train_dataset = NpzDataset(file_path, operation="train")
        valid_dataset = NpzDataset(file_path, operation="valid")
        test_dataset = NpzDataset(file_path, operation="test")

        train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset), shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

        return train_loader, valid_loader, test_loader

    def model_training_and_evaluation(self, model, weights, num_hidden_layers: int, verbose: bool):
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

    def pruning_model(self, model, weight_attr: str, set_weights_to_zero: bool):
        _, least_important_prune_indices = (
            model.pruning(pruning_percentage=self.cfg.get("subset_percentage").get(self.dataset_name),
                          pruning_method=self.cfg.get("pruning_method"))
        )

        if set_weights_to_zero:
            weight_tensor = getattr(model, weight_attr).data
            weight_tensor[:, least_important_prune_indices] = 0
            return model, least_important_prune_indices
        else:
            return least_important_prune_indices

    def prune_initial_model(self, model, set_weights_to_zero: bool):
        return self.pruning_model(model, weight_attr="alpha_weights", set_weights_to_zero=set_weights_to_zero)

    def prune_subsequent_model(self, model, set_weights_to_zero: bool):
        return self.pruning_model(model, weight_attr="extended_beta_weights", set_weights_to_zero=set_weights_to_zero)

    def prune_final_model(self, model, set_weights_to_zero: bool):
        return self.pruning_model(model, weight_attr="extended_gamma_weights", set_weights_to_zero=set_weights_to_zero)

    def create_train_prune_aux_model(self, model, model_type, weight_attr: str, least_important_prune_indices=None):
        net_cfg = self.get_network_config(model_type)
        aux_model = ModelFactory.create(model_type, net_cfg)

        aux_model.train_layer(self.train_loader)
        most_important_prune_indices, _ = (
            aux_model.pruning(pruning_percentage=self.cfg.get("subset_percentage").get(self.dataset_name),
                              pruning_method=self.cfg.get("pruning_method"))
        )

        best_weight_tensor = getattr(aux_model, weight_attr).data
        best_weights = best_weight_tensor[:, most_important_prune_indices]
        if least_important_prune_indices is None:
            least_important_prune_indices = self.prune_initial_model(model, set_weights_to_zero=False)

        weight_tensor = getattr(model, weight_attr).data
        weight_tensor[:, least_important_prune_indices] = best_weights

        return model

    def create_train_prune_initial_aux_model(self, model, model_type, least_important_prune_indices):
        return self.create_train_prune_aux_model(model,
                                                 model_type,
                                                 weight_attr="alpha_weights",
                                                 least_important_prune_indices=least_important_prune_indices)

    def create_train_prune_subsequent_aux_model(self, model, model_type, least_important_prune_indices):
        return self.create_train_prune_aux_model(model,
                                                 model_type,
                                                 weight_attr="extended_beta_weights",
                                                 least_important_prune_indices=least_important_prune_indices)

    def create_train_prune_final_aux_model(self, model, model_type, least_important_prune_indices):
        return self.create_train_prune_aux_model(model,
                                                 model_type,
                                                 weight_attr="extended_gamma_weights",
                                                 least_important_prune_indices=least_important_prune_indices)
    
    def main(self):
        # Load data
        accuracies = []
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
            accuracies.append((initial_model_training_metrics,
                               initial_model_testing_metrics,
                               initial_model_subs_weights_training_metrics,
                               initial_model_subs_weights_testing_metrics,
                               subsequent_model_subs_weights_training_metrics,
                               subsequent_model_subs_weights_testing_metrics,
                               subsequent_model_model_subs_weights_training_metrics,
                               subsequent_model_model_subs_weights_testing_metrics,
                               final_model_subs_weights_training_metrics,
                               final_model_subs_weights_testing_metrics,
                               final_model_model_subs_weights_training_metrics,
                               final_model_model_subs_weights_testing_metrics
                               ))

            best = final_model_subs_weights_testing_metrics if final_model_subs_weights_testing_metrics[0] > final_model_model_subs_weights_testing_metrics[0] else final_model_model_subs_weights_testing_metrics

            metrics = reorder_metrics_lists(train_metrics=final_model_model_subs_weights_training_metrics,
                                            test_metrics=best,
                                            training_time_list=training_time)
            insert_data_to_excel(self.save_filename, self.cfg.get("dataset_name"), i + 2, metrics, "ipmpdrnn")

            accuracies.clear()
            training_time.clear()

        average_columns_in_excel(self.save_filename)


if __name__ == "__main__":
    try:
        ipmpdrnn = Experiment()
        ipmpdrnn.main()
    except KeyboardInterrupt as kie:
        logging.error(f"Keyboard interrupt received: {kie}")
