import colorama
import os
import logging
import torch

from nn.dataloaders.npz_dataloader import NpzDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import general_dataset_configs, drnn_paths_config
from nn.models.mdprnn_model import MultiPhaseDeepRandomizedNeuralNetwork
from utils.utils import (average_columns_in_excel, create_timestamp, insert_data_to_excel, setup_logger,
                         load_config_json)


class Experiment:
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- __I N I T__ ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        # Create timestamp as the program begins to execute.
        timestamp = create_timestamp()

        # Initialize paths and settings
        setup_logger()
        self.cfg = (
            load_config_json(json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_ipmdrnn"),
                             json_filename=JSON_FILES_PATHS.get_data_path("config_ipmdrnn"))
        )
        self.gen_ds_cfg = general_dataset_configs(self.cfg.get("dataset_name"))
        drnn_config = drnn_paths_config(self.cfg.get("dataset_name"))

        self.filename = (
            os.path.join(
                drnn_config.get("ipmpdrnn").get("path_to_results"),
                f"{timestamp}_sp_{self.cfg.get('subset_percentage')}_pm_{self.cfg.get('pruning_method')}.xlsx")
        )

        if self.cfg.get("method") not in ["BASE", "EXP_ORT", "EXP_ORT_C"]:
            raise ValueError(f"Wrong method was given: {self.cfg.get('method')}")

        if self.cfg.get("seed"):
            torch.manual_seed(1234)

        # Load neurons
        self.first_layer_num_data = self.gen_ds_cfg.get("num_train_data")
        self.first_layer_input_nodes = self.gen_ds_cfg.get("num_features")
        self.first_layer_hidden_nodes = self.get_num_of_neurons(self.cfg.get("method"))[0]
        self.first_layer_output_nodes = self.gen_ds_cfg.get("num_classes")

        file_path = general_dataset_configs(self.cfg.get('dataset_name')).get("cached_dataset_file")
        self.train_loader, self.test_loader = self.create_train_test_datasets(file_path)

        colorama.init()

    def get_num_of_neurons(self, method):
        num_neurons = {
            "BASE": self.gen_ds_cfg.get("eq_neurons"),
            "EXP_ORT": self.gen_ds_cfg.get("exp_neurons"),
            "EXP_ORT_C": self.gen_ds_cfg.get("exp_neurons"),
        }
        return num_neurons[method]

    @staticmethod
    def create_train_test_datasets(file_path):
        train_dataset = NpzDataset(file_path, operation="train")
        test_dataset = NpzDataset(file_path, operation="test")

        train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

        return train_loader, test_loader

    @staticmethod
    def calc_ort(x):
        x = x.reshape(x.shape[0], 1)
        q, _ = torch.linalg.qr(x)
        return q

    def change_ort(self, model, least_important_prune_indices):
        worst_weights = model.alpha_weights.data[:, least_important_prune_indices]
        ort_list = [self.calc_ort(row) for row in worst_weights]
        ort = torch.stack(ort_list, dim=0)
        return ort.squeeze(-1)

    def random_weights(self, model, least_important_prune_indices, train_loader, test_loader):
        num_neurons_to_prune = int(self.cfg.get("subset_percentage") * model.beta_weights.shape[0])
        best_weights = torch.randn((self.first_layer_input_nodes, num_neurons_to_prune))
        model.alpha_weights.data[:, least_important_prune_indices] = best_weights
        model.train_first_layer(train_loader)
        training_acc = model.predict_and_evaluate(train_loader, "train")
        testing_acc = model.predict_and_evaluate(test_loader, "test")

        return training_acc, testing_acc

    def initial_model_training(self, initial_model):
        initial_model.train_first_layer(self.train_loader)
        tr1acc = initial_model.predict_and_evaluate(self.train_loader, "train")
        te1acc = initial_model.predict_and_evaluate(self.test_loader, "test")

        return initial_model, tr1acc, te1acc

    def prune_initial_model(self, initial_model, set_weights_to_zero: bool):
        _, least_important_prune_indices = (
            initial_model.pruning(pruning_percentage=self.cfg.get("subset_percentage"),
                                  pruning_method=self.cfg.get("pruning_method"))
        )

        if set_weights_to_zero:
            initial_model.alpha_weights.data[:, least_important_prune_indices] = 0
            return initial_model, least_important_prune_indices
        else:
            return least_important_prune_indices

    def create_train_prune_aux_model(self, initial_model, least_important_prune_indices=None):
        aux_model = MultiPhaseDeepRandomizedNeuralNetwork(num_data=self.first_layer_num_data,
                                                          num_features=self.first_layer_input_nodes,
                                                          hidden_nodes=self.first_layer_hidden_nodes,
                                                          output_nodes=self.first_layer_output_nodes,
                                                          activation_function=self.cfg.get('activation'))

        aux_model.train_first_layer(self.train_loader)
        most_important_prune_indices, _ = aux_model.pruning(pruning_percentage=self.cfg.get("subset_percentage"),
                                                            pruning_method=self.cfg.get("pruning_method"))

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
            initial_model = MultiPhaseDeepRandomizedNeuralNetwork(num_data=self.first_layer_num_data,
                                                                  num_features=self.first_layer_input_nodes,
                                                                  hidden_nodes=self.first_layer_hidden_nodes,
                                                                  output_nodes=self.first_layer_output_nodes,
                                                                  activation_function=self.cfg.get('activation'))

            # Train and evaluate the model
            initial_model, initial_model_training_acc, initial_model_testing_acc = (
                self.initial_model_training(initial_model)
            )

            # Pruning
            initial_model, least_important_prune_indices = (
                self.prune_initial_model(initial_model, set_weights_to_zero=True)
            )

            # Train and evaluate network again with pruned weights
            initial_model, initial_model_pruned_training_acc, initial_model_pruned_testing_acc = (
                self.initial_model_training(initial_model)
            )

            # Create aux model 1
            initial_model = self.create_train_prune_aux_model(initial_model, least_important_prune_indices)
            initial_model, initial_model_subs_weights_training_acc, initial_model_subs_weights_testing_acc = (
                self.initial_model_training(initial_model)
            )

            # Create aux model 2
            initial_model = self.create_train_prune_aux_model(initial_model, least_important_prune_indices)
            initial_model, initial_model_subs_weights_training_acc_2, initial_model_subs_weights_testing_acc_2 = (
                self.initial_model_training(initial_model)
            )

            # Excel
            accuracies.append((initial_model_training_acc, initial_model_testing_acc,
                               initial_model_pruned_training_acc, initial_model_pruned_testing_acc,
                               initial_model_subs_weights_training_acc, initial_model_subs_weights_testing_acc,
                               initial_model_subs_weights_training_acc_2, initial_model_subs_weights_testing_acc_2))

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
