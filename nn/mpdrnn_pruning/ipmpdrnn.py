import colorama
import os
import logging
import torch

from nn.dataloaders.npz_dataloader import NpzDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import general_dataset_configs, drnn_paths_config
from nn.models.mpdrnn_model import MultiPhaseDeepRandomizedNeuralNetworkBase
from utils.utils import (average_columns_in_excel, create_timestamp, insert_data_to_excel, setup_logger,
                         load_config_json)


class Experiment:
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- __I N I T__ ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        # Create timestamp as the program begins to execute.
        timestamp = create_timestamp()

        setup_logger()

        # Initialize paths and settings
        self.cfg = (
            load_config_json(json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_ipmdrnn"),
                             json_filename=JSON_FILES_PATHS.get_data_path("config_ipmdrnn"))
        )

        self.dataset_name = self.cfg.get("dataset_name")

        self.gen_ds_cfg = general_dataset_configs(self.cfg.get("dataset_name"))
        drnn_config = drnn_paths_config(self.cfg.get("dataset_name"))

        self.filename = (
            os.path.join(
                drnn_config.get("ipmpdrnn").get("path_to_results"),
                f"{timestamp}"
                f"_sp_{self.cfg.get('subset_percentage').get(self.dataset_name)}"
                f"_pm_{self.cfg.get('pruning_method')}"
                f"_rcond_{self.cfg.get('rcond').get(self.dataset_name)}.xlsx")
        )

        if self.cfg.get("method") not in ["BASE", "EXP_ORT", "EXP_ORT_C"]:
            raise ValueError(f"Wrong method was given: {self.cfg.get('method')}")

        if self.cfg.get("seed"):
            torch.manual_seed(1234)

        # Load neurons
        self.first_layer_num_data = self.gen_ds_cfg.get("num_train_data")
        self.first_layer_input_nodes = self.gen_ds_cfg.get("num_features")
        self.first_layer_hidden_nodes = self.get_num_of_neurons(self.cfg.get("method"), self.dataset_name)
        self.first_layer_output_nodes = self.gen_ds_cfg.get("num_classes")

        file_path = general_dataset_configs(self.cfg.get('dataset_name')).get("cached_dataset_file")
        self.train_loader, self.valid_loader, self.test_loader = (
            self.create_train_valid_test_datasets(file_path)
        )

        colorama.init()

    def get_num_of_neurons(self, method, dataset_name):
        num_neurons = {
            "BASE": self.cfg.get("eq_neurons").get(dataset_name),
            "EXP_ORT": self.cfg.get("exp_neurons").get(dataset_name),
            "EXP_ORT_C": self.cfg.get("exp_neurons").get(dataset_name),
        }
        return num_neurons[method]

    @staticmethod
    def create_train_valid_test_datasets(file_path):
        train_dataset = NpzDataset(file_path, operation="train")
        valid_dataset = NpzDataset(file_path, operation="valid")
        test_dataset = NpzDataset(file_path, operation="test")

        train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset), shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

        return train_loader, valid_loader, test_loader

    def initial_model_training_and_eval(self, initial_model):
        initial_model.train_layer(self.train_loader)

        train_acc = (
            initial_model.predict_and_evaluate(
                dataloader=self.train_loader,
                operation="train",
                layer_weights=initial_model.beta_weights,
                num_hidden_layers=1,
                verbose=True
            )
        )
        test_acc = (
            initial_model.predict_and_evaluate(
                dataloader=self.test_loader,
                operation="test",
                layer_weights=initial_model.beta_weights,
                num_hidden_layers=1,
                verbose=True
            )
        )

        return initial_model, train_acc, test_acc

    def prune_initial_model(self, initial_model, set_weights_to_zero: bool):
        _, least_important_prune_indices = (
            initial_model.pruning(pruning_percentage=self.cfg.get("subset_percentage").get(self.dataset_name),
                                  pruning_method=self.cfg.get("pruning_method"))
        )

        if set_weights_to_zero:
            initial_model.alpha_weights.data[:, least_important_prune_indices] = 0
            return initial_model, least_important_prune_indices
        else:
            return least_important_prune_indices

    def create_train_prune_aux_model(self, initial_model, least_important_prune_indices=None):
        aux_model = (
            MultiPhaseDeepRandomizedNeuralNetworkBase(
                num_data=self.first_layer_num_data,
                num_features=self.first_layer_input_nodes,
                hidden_nodes=self.first_layer_hidden_nodes,
                output_nodes=self.first_layer_output_nodes,
                activation_function=self.cfg.get('activation'),
                method=self.cfg.get('method'),
                rcond=self.cfg.get('rcond').get(self.dataset_name),
            )
        )

        aux_model.train_layer(self.train_loader)
        most_important_prune_indices, _ = (
            aux_model.pruning(pruning_percentage=self.cfg.get("subset_percentage").get(self.dataset_name),
                              pruning_method=self.cfg.get("pruning_method"))
        )

        best_weights = aux_model.alpha_weights.data[:, most_important_prune_indices]
        if least_important_prune_indices is None:
            least_important_prune_indices = self.prune_initial_model(initial_model, set_weights_to_zero=False)
        initial_model.alpha_weights.data[:, least_important_prune_indices] = best_weights

        return initial_model

    def main(self):
        # Load data
        accuracies = []

        for i in tqdm(range(self.cfg.get('number_of_tests')), desc=colorama.Fore.CYAN + "Process"):
            # Create model
            initial_model = (
                MultiPhaseDeepRandomizedNeuralNetworkBase(
                    num_data=self.first_layer_num_data,
                    num_features=self.first_layer_input_nodes,
                    hidden_nodes=self.first_layer_hidden_nodes,
                    output_nodes=self.first_layer_output_nodes,
                    activation_function=self.cfg.get('activation'),
                    method=self.cfg.get('method'),
                    rcond=self.cfg.get('rcond').get(self.dataset_name),
                )
            )

            # Train and evaluate the model
            initial_model, initial_model_training_acc, initial_model_testing_acc = (
                self.initial_model_training_and_eval(initial_model)
            )

            # Pruning
            least_important_prune_indices = (
                self.prune_initial_model(initial_model, set_weights_to_zero=False)
            )

            # Create aux model 1
            initial_model = (
                self.create_train_prune_aux_model(initial_model=initial_model,
                                                  least_important_prune_indices=least_important_prune_indices)
            )

            initial_model, initial_model_subs_weights_training_acc, initial_model_subs_weights_testing_acc = (
                self.initial_model_training_and_eval(initial_model=initial_model)
            )

            # Excel
            accuracies.append((initial_model_training_acc[0], initial_model_testing_acc[0],
                               initial_model_subs_weights_training_acc[0], initial_model_subs_weights_testing_acc[0]))

            insert_data_to_excel(filename=self.filename,
                                 dataset_name=self.cfg.get("dataset_name"),
                                 row=i + 2,
                                 data=accuracies,
                                 network="ipmpdrnn")

            accuracies.clear()

        average_columns_in_excel(self.filename)


if __name__ == "__main__":
    try:
        ipmpdrnn = Experiment()
        ipmpdrnn.main()
    except KeyboardInterrupt as kie:
        logging.error(f"Keyboard interrupt received: {kie}")
