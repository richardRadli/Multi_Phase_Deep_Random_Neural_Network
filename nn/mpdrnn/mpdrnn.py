import colorama
import os
import torch

from tqdm import tqdm

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import general_dataset_configs, drnn_paths_config
from nn.models.model_selector import ModelFactory
from utils.utils import (average_columns_in_excel, create_timestamp, create_train_valid_test_datasets,
                         insert_data_to_excel, load_config_json, reorder_metrics_lists, get_num_of_neurons)


class MPDRNN:
    def __init__(self):
        # Create timestamp as the program begins to execute.
        timestamp = create_timestamp()

        # Initialize paths and settings
        self.cfg = (
            load_config_json(json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_mpdrnn"),
                             json_filename=JSON_FILES_PATHS.get_data_path("config_mpdrnn"))
        )

        # Boilerplate
        self.dataset_name = self.cfg.get("dataset_name")
        self.method = self.cfg.get('method')
        self.penalty_term = self.cfg.get('penalty')
        self.rcond = self.cfg.get("rcond").get(self.cfg.get("method"))
        self.activation = self.cfg.get('activation')
        self.gen_ds_cfg = general_dataset_configs(self.cfg.get("dataset_name"))
        drnn_config = drnn_paths_config(self.dataset_name)

        self.initial_model = None
        self.subsequent_model = None

        self.filename = (
            os.path.join(
                drnn_config.get("mpdrnn").get("path_to_results"),
                f"{timestamp}_{self.dataset_name}_dataset_{self.method}_method_{self.penalty_term}_penalty.xlsx"
            )
        )

        if self.cfg.get("method") not in ["BASE", "EXP_ORT", "EXP_ORT_C"]:
            raise ValueError(f"Wrong method was given: {self.cfg.get('method')}")

        if self.cfg.get("seed"):
            torch.manual_seed(1234)

        file_path = general_dataset_configs(self.dataset_name).get("cached_dataset_file")
        self.train_loader, _, self.test_loader = create_train_valid_test_datasets(file_path)

        colorama.init()

    def get_network_config(self, network_type):
        net_cfg = {
            "MultiPhaseDeepRandomizedNeuralNetworkBase": {
                "first_layer_num_data": self.gen_ds_cfg.get("num_train_data"),
                "first_layer_num_features": self.gen_ds_cfg.get("num_features"),
                "first_layer_num_hidden": get_num_of_neurons(self.cfg, self.method),
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

    def model_training_and_evaluation(self, model, weights, num_hidden_layers, verbose):
        model.train_layer(self.train_loader)

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

    def main(self):
        training_time = []

        for i in tqdm(range(self.cfg.get('number_of_tests')), desc=colorama.Fore.CYAN + "Process"):
            net_cfg = self.get_network_config("MultiPhaseDeepRandomizedNeuralNetworkBase")
            self.initial_model = ModelFactory.create("MultiPhaseDeepRandomizedNeuralNetworkBase", net_cfg)

            self.initial_model, initial_model_training_metrics, initial_model_testing_metrics = (
                self.model_training_and_evaluation(model=self.initial_model,
                                                   weights=self.initial_model.beta_weights,
                                                   num_hidden_layers=1,
                                                   verbose=True)
            )

            training_time.append(self.initial_model.train_ith_layer.execution_time)

            # Subsequent Model
            net_cfg = self.get_network_config("MultiPhaseDeepRandomizedNeuralNetworkSubsequent")
            self.subsequent_model = ModelFactory.create("MultiPhaseDeepRandomizedNeuralNetworkSubsequent", net_cfg)

            self.subsequent_model, subsequent_model_training_metrics, subsequent_model_testing_metrics = (
                self.model_training_and_evaluation(model=self.subsequent_model,
                                                   weights=[self.subsequent_model.extended_beta_weights,
                                                            self.subsequent_model.gamma_weights],
                                                   num_hidden_layers=2,
                                                   verbose=True)
            )

            training_time.append(self.subsequent_model.train_ith_layer.execution_time)

            net_cfg = self.get_network_config(network_type="MultiPhaseDeepRandomizedNeuralNetworkFinal")
            final_model = ModelFactory.create("MultiPhaseDeepRandomizedNeuralNetworkFinal", net_cfg)

            final_model, final_model_training_metrics, final_model_testing_metrics = (
                self.model_training_and_evaluation(model=final_model,
                                                   weights=[final_model.extended_beta_weights,
                                                            final_model.extended_gamma_weights,
                                                            final_model.delta_weights],
                                                   num_hidden_layers=3,
                                                   verbose=True)
            )

            training_time.append(final_model.train_ith_layer.execution_time)

            metrics = reorder_metrics_lists(train_metrics=final_model_training_metrics,
                                            test_metrics=final_model_testing_metrics,
                                            training_time_list=training_time)
            insert_data_to_excel(self.filename, self.cfg.get("dataset_name"), i + 2, metrics)
            training_time.clear()

        average_columns_in_excel(self.filename)


if __name__ == "__main__":
    mpdrnn = MPDRNN()
    mpdrnn.main()
