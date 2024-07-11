import colorama
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from nn.dataloader.npz_dataloader import NpzDataset
from torch.utils.data import DataLoader  # Subset
from tqdm import tqdm

from config.config import MPDRNNConfig
from config.dataset_config import elm_general_dataset_configs
from nn.models.mdprnn_model import MultiPhaseDeepRandomizedNeuralNetwork
from utils.utils import (average_columns_in_excel, create_timestamp, display_dataset_info, insert_data_to_excel,
                         setup_logger)


class Experiment:
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- __I N I T__ ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        timestamp = create_timestamp()
        self.filename = os.path.join("C:/Users/ricsi/Desktop/results", f"{timestamp}.xlsx")

        # Initialize paths and settings
        setup_logger()
        self.cfg = MPDRNNConfig().parse()
        self.gen_ds_cfg = elm_general_dataset_configs(self.cfg)
        display_dataset_info(self.gen_ds_cfg)

        if self.cfg.method not in ["BASE", "EXP_ORT", "EXP_ORT_C"]:
            raise ValueError(f"Wrong method was given: {self.cfg.method}")

        if self.cfg.seed:
            torch.manual_seed(1234)

        self.method = self.cfg.method
        self.activation = self.cfg.activation

        # Load neurons
        self.first_layer_num_data = self.gen_ds_cfg.get("num_train_data")
        self.first_layer_input_nodes = self.gen_ds_cfg.get("num_features")
        self.first_layer_hidden_nodes = self.get_num_of_neurons(self.method)[0]
        self.first_layer_output_nodes = self.gen_ds_cfg.get("num_classes")

        file_path = elm_general_dataset_configs(self.cfg).get("cached_dataset_file")
        self.train_loader, self.test_loader = self.create_datasets_full(file_path)

        colorama.init()

    def get_num_of_neurons(self, method):
        num_neurons = {
            "BASE": self.gen_ds_cfg.get("eq_neurons"),
            "EXP_ORT": self.gen_ds_cfg.get("exp_neurons"),
            "EXP_ORT_C": self.gen_ds_cfg.get("exp_neurons"),
        }
        return num_neurons[method]

    # @staticmethod
    # def create_subset_percentage(dataset, percentage):
    #     total_samples = len(dataset)
    #     subset_size = int(total_samples * percentage)
    #     indices = list(range(total_samples))
    #
    #     selected_indices = indices[:subset_size]
    #     subset = Subset(dataset, selected_indices)
    #
    #     return subset

    # def create_datasets_split(self, file_path, subset_percentage):
    #     full_train_dataset = NpzDataset(file_path, operation="train")
    #     test_dataset = NpzDataset(file_path, operation="test")
    #
    #     train_dataset = self.create_subset(full_train_dataset, subset_percentage)
    #
    #     train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
    #     test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    #
    #     return train_loader, test_loader

    # @staticmethod
    # def create_subsets(dataset, num_splits):
    #     total_samples = len(dataset)
    #     indices = list(range(total_samples))
    #     split_size = total_samples // num_splits
    #
    #     subset = []
    #     for i in range(num_splits):
    #         start_idx = i * split_size
    #         if i == num_splits - 1:
    #             end_idx = total_samples
    #         else:
    #             end_idx = (i + 1) * split_size
    #         subset_indices = indices[start_idx:end_idx]
    #         subset.append(Subset(dataset, subset_indices))
    #
    #     return subset

    # def create_datasets(self, file_path):
    #     full_train_dataset = NpzDataset(file_path, operation="train")
    #     test_dataset = NpzDataset(file_path, operation="test")
    #
    #     num_splits = 5
    #     train_subsets = self.create_subsets(full_train_dataset, num_splits)
    #
    #     # Create data loaders for each subset
    #     train_loaders = [DataLoader(dataset=subset, batch_size=len(subset), shuffle=False) for subset in train_subsets]
    #     test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    #
    #     return train_loaders, test_loader

    @staticmethod
    def replace_zeros_with_random(module, param_name):
        param = getattr(module, param_name)
        mask = param == 0
        random_values = torch.FloatTensor(np.random.uniform(-1, 1, mask.sum().item()))
        param.data[mask] = random_values

    @staticmethod
    def check_for_zeros(module, param_name):
        param = getattr(module, param_name)
        if torch.any(param == 0):
            print(f"Parameter {param_name} in module {module} contains zeros.")
        else:
            print(f"Parameter {param_name} in module {module} does not contain any zeros.")

    @staticmethod
    def visualize_weights(pruned_weights, title):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(pruned_weights.flatten(), bins=50, color='blue', alpha=0.7)
        plt.title(f'Histogram of Alpha Weights {title} pruning')
        plt.xlabel('Weight Values')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def create_datasets_full(file_path):
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
        num_neurons_to_prune = int(self.cfg.subset_percentage * model.beta_weights.shape[0])
        best_weights = torch.randn((self.first_layer_input_nodes, num_neurons_to_prune))
        model.alpha_weights.data[:, least_important_prune_indices] = best_weights
        model.train_first_layer(train_loader)
        training_acc = model.predict_and_evaluate(train_loader, "train")
        testing_acc = model.predict_and_evaluate(test_loader, "test")

        return training_acc, testing_acc

    def base_model_training(self, base_model):
        base_model.train_first_layer(self.train_loader)
        tr1acc = base_model.predict_and_evaluate(self.train_loader, "train")
        te1acc = base_model.predict_and_evaluate(self.test_loader, "test")

        return base_model, tr1acc, te1acc

    def prune_base_model(self, base_model):
        _, least_important_prune_indices = (
            base_model.pruning(pruning_percentage=self.cfg.subset_percentage,
                               pruning_method=self.cfg.pruning_method)
        )
        base_model.alpha_weights.data[:, least_important_prune_indices] = 0

        return base_model, least_important_prune_indices

    def create_train_prune_aux_model(self, base_model, least_important_prune_indices):
        # Generate new alpha weights
        aux_model = MultiPhaseDeepRandomizedNeuralNetwork(num_data=self.first_layer_num_data,
                                                          num_features=self.first_layer_input_nodes,
                                                          hidden_nodes=self.first_layer_hidden_nodes,
                                                          output_nodes=self.first_layer_output_nodes,
                                                          activation_function="leaky_ReLU")

        aux_model.train_first_layer(self.train_loader)
        most_important_prune_indices, _ = aux_model.pruning(pruning_percentage=self.cfg.subset_percentage,
                                                            pruning_method=self.cfg.pruning_method)

        best_weights = aux_model.alpha_weights.data[:, most_important_prune_indices]
        base_model.alpha_weights.data[:, least_important_prune_indices] = best_weights

        return base_model

    def main_full_sets(self):
        # Load data
        accuracies = []

        for i in tqdm(range(self.cfg.number_of_tests), desc=colorama.Fore.CYAN + "Process"):
            # Create model
            base_model = MultiPhaseDeepRandomizedNeuralNetwork(num_data=self.first_layer_num_data,
                                                               num_features=self.first_layer_input_nodes,
                                                               hidden_nodes=self.first_layer_hidden_nodes,
                                                               output_nodes=self.first_layer_output_nodes,
                                                               activation_function="leaky_ReLU")

            # Train and evaluate the model
            base_model, base_model_training_acc, base_model_testing_acc = (
                self.base_model_training(base_model)
            )

            # Pruning
            base_model, least_important_prune_indices = self.prune_base_model(base_model)

            # Train and evaluate network again with pruned weights
            base_model, base_model_pruned_training_acc, base_model_pruned_testing_acc = (
                self.base_model_training(base_model)
            )

            # Create aux model
            base_model = self.create_train_prune_aux_model(base_model, least_important_prune_indices)
            base_model, base_model_substituted_weights_training_acc, base_model_substituted_weights_testing_acc = (
                self.base_model_training(base_model)
            )

            base_model = self.create_train_prune_aux_model(base_model, least_important_prune_indices)
            base_model, base_model_substituted_weights_training_acc_2, base_model_substituted_weights_testing_acc_2 = (
                self.base_model_training(base_model)
            )

            # aux_model_3 = MultiPhaseDeepRandomizedNeuralNetwork(num_data=self.first_layer_num_data,
            #                                                     num_features=self.first_layer_input_nodes,
            #                                                     hidden_nodes=self.first_layer_hidden_nodes,
            #                                                     output_nodes=self.first_layer_output_nodes,
            #                                                     activation_function="leaky_ReLU")
            #
            # aux_model_3.train_first_layer(train_loader)
            # most_important_prune_indices, _ = aux_model_3.pruning(pruning_percentage=self.cfg.subset_percentage,
            #                                                       pruning_method=self.cfg.pruning_method)
            #
            # _, least_important_prune_indices = base_model.pruning(pruning_percentage=self.cfg.subset_percentage,
            #                                                       pruning_method=self.cfg.pruning_method)
            #
            # best_weights = aux_model_3.alpha_weights.data[:, most_important_prune_indices]
            # base_model.alpha_weights.data[:, least_important_prune_indices] = best_weights
            #
            # base_model.train_first_layer(train_loader)
            # tr5acc = base_model.predict_and_evaluate(train_loader, "train")
            # te5acc = base_model.predict_and_evaluate(test_loader, "test")

            # Excel
            accuracies.append((base_model_training_acc, base_model_testing_acc,
                               base_model_pruned_training_acc, base_model_pruned_testing_acc,
                               base_model_substituted_weights_training_acc, base_model_substituted_weights_testing_acc,
                               base_model_substituted_weights_training_acc_2,
                               base_model_substituted_weights_testing_acc_2))
            insert_data_to_excel(filename=self.filename,
                                 dataset_name=self.cfg.dataset_name,
                                 row=i + 2,
                                 data=accuracies)
            average_columns_in_excel(self.filename)
            accuracies.clear()


if __name__ == "__main__":
    mpdrnn = Experiment()
    mpdrnn.main_full_sets()
