import colorama
import os
import torch

from nn.dataloaders.npz_dataloader import NpzDataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import general_dataset_configs, drnn_paths_config
from nn.models.mdprnn_model import MultiPhaseDeepRandomizedNeuralNetwork
from utils.utils import (average_columns_in_excel, create_timestamp, insert_data_to_excel, setup_logger,
                         load_config_json)


class Experiment2:
    def __init__(self):
        # Create timestamp as the program begins to execute.
        timestamp = create_timestamp()

        # Initialize paths and settings
        setup_logger()
        self.cfg = load_config_json(json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_ipmdrnn"),
                                    json_filename=JSON_FILES_PATHS.get_data_path("config_ipmdrnn"),)
        self.gen_ds_cfg = general_dataset_configs(self.cfg.get("dataset_name"))
        drnn_cfg = drnn_paths_config(self.cfg.get("dataset_name"))

        self.save_filename = (
            os.path.join(
                drnn_cfg.get("ipmpdrnn").get("path_to_results"),
                f"{timestamp}_partition_dataset_sp_{self.cfg.get('subset_percentage')}_pm_"
                f"{self.cfg.get('pruning_method')}.xlsx")
        )

        if self.cfg.get("method") not in ["BASE", "EXP_ORT", "EXP_ORT_C"]:
            raise ValueError(f"Wrong method was given: {self.cfg.get('method')}")

        if self.cfg.get("seed"):
            torch.manual_seed(1234)

        # Load neurons
        self.first_layer_input_nodes = self.gen_ds_cfg.get("num_features")
        self.first_layer_hidden_nodes = self.get_num_of_neurons(self.cfg.get("method"))[0]
        self.first_layer_output_nodes = self.gen_ds_cfg.get("num_classes")

        file_path = general_dataset_configs(self.cfg.get("dataset_name")).get("cached_dataset_file")
        self.sub_train_loader = self.create_datasets_subs(file_path, 3)
        self.train_loader, self.test_loader = self.create_dataset_full(file_path)

        colorama.init()

    def get_num_of_neurons(self, method):
        num_neurons = {
            "BASE": self.gen_ds_cfg.get("eq_neurons"),
            "EXP_ORT": self.gen_ds_cfg.get("exp_neurons"),
            "EXP_ORT_C": self.gen_ds_cfg.get("exp_neurons"),
        }
        return num_neurons[method]

    @staticmethod
    def create_subsets(dataset, num_splits):
        total_samples = len(dataset)
        indices = list(range(total_samples))
        split_size = total_samples // num_splits

        subset = []
        for i in range(num_splits):
            start_idx = i * split_size
            if i == num_splits - 1:
                end_idx = total_samples
            else:
                end_idx = (i + 1) * split_size
            subset_indices = indices[start_idx:end_idx]
            subset.append(Subset(dataset, subset_indices))

        return subset

    def create_datasets_subs(self, file_path, num_splits):
        full_train_dataset = NpzDataset(file_path, operation="train")
        train_subsets = self.create_subsets(full_train_dataset, num_splits)

        return [DataLoader(dataset=subset, batch_size=len(subset), shuffle=False) for subset in train_subsets]

    @staticmethod
    def create_dataset_full(file_path):
        train_dataset = NpzDataset(file_path, operation="train")
        test_dataset = NpzDataset(file_path, operation="test")

        train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

        return train_loader, test_loader

    def initial_model_training(self, initial_model, train_loader):
        initial_model.train_first_layer(train_loader)
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

    def create_train_prune_aux_model(self, initial_model, train_loader, least_important_prune_indices=None):
        first_layer_num_data = train_loader.batch_size
        # Generate new alpha weights
        aux_model = MultiPhaseDeepRandomizedNeuralNetwork(num_data=first_layer_num_data,
                                                          num_features=self.first_layer_input_nodes,
                                                          hidden_nodes=self.first_layer_hidden_nodes,
                                                          output_nodes=self.first_layer_output_nodes,
                                                          activation_function="leaky_ReLU")

        aux_model.train_first_layer(train_loader)
        most_important_prune_indices, _ = aux_model.pruning(pruning_percentage=self.cfg.get("subset_percentage"),
                                                            pruning_method=self.cfg.get("pruning_method"))

        best_weights = aux_model.alpha_weights.data[:, most_important_prune_indices]
        if least_important_prune_indices is None:
            least_important_prune_indices = self.prune_initial_model(initial_model, set_weights_to_zero=False)
        initial_model.alpha_weights.data[:, least_important_prune_indices] = best_weights

        return initial_model

    def main(self):
        accuracies = []

        for i in tqdm(range(self.cfg.get("number_of_tests")), desc=colorama.Fore.CYAN + "Process"):
            train_loader_1 = self.sub_train_loader[0]
            train_loader_2 = self.sub_train_loader[1]
            train_loader_3 = self.sub_train_loader[2]

            first_layer_num_data = train_loader_1.batch_size
            initial_model = MultiPhaseDeepRandomizedNeuralNetwork(num_data=first_layer_num_data,
                                                                  num_features=self.first_layer_input_nodes,
                                                                  hidden_nodes=self.first_layer_hidden_nodes,
                                                                  output_nodes=self.first_layer_output_nodes,
                                                                  activation_function="leaky_ReLU")

            initial_model, base_model_training_acc, base_model_testing_acc = (
                self.initial_model_training(initial_model, train_loader_1)
            )

            # Pruning
            initial_model, least_important_prune_indices = (
                self.prune_initial_model(initial_model,
                                         set_weights_to_zero=True)
            )

            # Train and evaluate network again with pruned weights
            initial_model, base_model_pruned_training_acc, base_model_pruned_testing_acc = (
                self.initial_model_training(initial_model, train_loader_1)
            )

            # Create aux model
            initial_model = self.create_train_prune_aux_model(initial_model, train_loader_2, least_important_prune_indices)
            initial_model, base_model_substituted_weights_training_acc, base_model_substituted_weights_testing_acc = (
                self.initial_model_training(initial_model, train_loader_2)
            )

            # Create aux model
            initial_model = self.create_train_prune_aux_model(initial_model, train_loader_3)
            initial_model, base_model_substituted_weights_training_acc_2, base_model_substituted_weights_testing_acc_2 = (
                self.initial_model_training(initial_model, train_loader_3)
            )

            accuracies.append((base_model_training_acc, base_model_testing_acc,
                               base_model_pruned_training_acc, base_model_pruned_testing_acc,
                               base_model_substituted_weights_training_acc, base_model_substituted_weights_testing_acc,
                               base_model_substituted_weights_training_acc_2, base_model_substituted_weights_testing_acc_2))
            insert_data_to_excel(filename=self.save_filename,
                                 dataset_name=self.cfg.get("dataset_name"),
                                 row=i + 2,
                                 data=accuracies)
            accuracies.clear()

        average_columns_in_excel(self.save_filename)


if __name__ == "__main__":
    exp2 = Experiment2()
    exp2.main()
