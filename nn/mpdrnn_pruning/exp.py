import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from nn.dataloader.npz_dataloader import NpzDataset
from torch.utils.data import DataLoader, Subset

from config.config import MPDRNNConfig
from config.dataset_config import elm_general_dataset_configs
from nn.models.mdprnn_model import MultiPhaseDeepRandomizedNeuralNetwork
from utils.utils import display_dataset_info, setup_logger


class Experiment:
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- __I N I T__ ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        # Initialize paths and settings
        setup_logger()
        self.cfg = MPDRNNConfig().parse()
        self.gen_ds_cfg = elm_general_dataset_configs(self.cfg)
        display_dataset_info(self.gen_ds_cfg)

        if self.cfg.method not in ["BASE", "EXP_ORT", "EXP_ORT_C"]:
            raise ValueError(f"Wrong method was given: {self.cfg.method}")

        if self.cfg.seed:
            torch.manual_seed(42)

        self.method = self.cfg.method
        self.activation = self.cfg.activation

        # Load data
        file_path = elm_general_dataset_configs(self.cfg).get("cached_dataset_file")
        self.train_loader, self.test_loader = (
            self.create_datasets(file_path, subset_percentage=self.cfg.subset_percentage)
        )

        # Load neurons
        self.first_layer_num_data = self.gen_ds_cfg.get("num_train_data")
        self.first_layer_input_nodes = self.gen_ds_cfg.get("num_features")
        self.first_layer_hidden_nodes = self.get_num_of_neurons(self.method)[0]
        self.first_layer_output_nodes = self.gen_ds_cfg.get("num_classes")

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
                end_idx = (i+1) * split_size
            subset_indices = indices[start_idx:end_idx]
            subset.append(Subset(dataset, subset_indices))

        return subset

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

    def create_datasets(self, file_path, subset_percentage):
        full_train_dataset = NpzDataset(file_path, operation="train")
        test_dataset = NpzDataset(file_path, operation="test")

        num_splits = 5
        train_subsets = self.create_subsets(full_train_dataset, num_splits)

        # Create data loaders for each subset
        train_loaders = [DataLoader(dataset=subset, batch_size=len(subset), shuffle=False) for subset in train_subsets]
        test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

        return train_loaders, test_loader

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
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    def main_subsets(self):
        model = MultiPhaseDeepRandomizedNeuralNetwork(num_data=self.first_layer_num_data,
                                                      num_features=self.first_layer_input_nodes,
                                                      hidden_nodes=self.first_layer_hidden_nodes,
                                                      output_nodes=self.first_layer_output_nodes,
                                                      activation_function="leaky_ReLU")

        alpha_weights_list = []

        for train_loader in self.train_loader:
            weights = model.alpha_weights.detach().numpy()
            self.visualize_weights(weights, "before")

            file_path = elm_general_dataset_configs(self.cfg).get("cached_dataset_file")
            train_loader, test_loader = self.create_datasets_full(file_path)

            model.train_first_layer(train_loader)
            model.predict_and_evaluate(train_loader, "train")
            model.predict_and_evaluate(test_loader, "test")

            model.prune_alpha_weights(0.2)

            pruned_alpha_weights = model.alpha_weights.detach().numpy()
            self.visualize_weights(pruned_alpha_weights, "after")

            # prune.l1_unstructured(model, name="alpha_weights", amount=0.2)
            # prune.remove(model, name="alpha_weights")

            model.train_first_layer(train_loader)
            model.predict_and_evaluate(train_loader, "train")
            model.predict_and_evaluate(test_loader, "test")

            model.alpha_weights = (
                nn.Parameter(torch.randn(self.first_layer_input_nodes, self.first_layer_hidden_nodes),
                             requires_grad=True)
            )
            new_alpha_weights = model.alpha_weights.detach().numpy()

            subs_percentage = 0.2
            least_imp_old, most_imp_old = self.find_weights_importance(pruned_alpha_weights, subs_percentage)
            least_imp_new, most_imp_new = self.find_weights_importance(new_alpha_weights, subs_percentage)
            substituted_weights = self.substitute_weights(pruned_alpha_weights, new_alpha_weights, least_imp_old,
                                                          most_imp_new)

            model.alpha_weights = nn.Parameter(torch.from_numpy(substituted_weights))
            model.train_first_layer(train_loader)
            model.predict_and_evaluate(train_loader, "train")
            model.predict_and_evaluate(test_loader, "test")

            weights = model.alpha_weights.detach().numpy()
            self.visualize_weights(weights, "after2")

    @staticmethod
    def create_datasets_full(file_path):
        train_dataset = NpzDataset(file_path, operation="train")
        test_dataset = NpzDataset(file_path, operation="test")

        train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

        return train_loader, test_loader

    def main_full_sets(self):
        # Load data
        file_path = elm_general_dataset_configs(self.cfg).get("cached_dataset_file")
        train_loader, test_loader = self.create_datasets_full(file_path)

        # Create model
        model = MultiPhaseDeepRandomizedNeuralNetwork(num_data=self.first_layer_num_data,
                                                      num_features=self.first_layer_input_nodes,
                                                      hidden_nodes=self.first_layer_hidden_nodes,
                                                      output_nodes=self.first_layer_output_nodes,
                                                      activation_function="leaky_ReLU")

        # Visualize original alpha weights
        alpha_weights = model.alpha_weights.detach().numpy()
        self.visualize_weights(alpha_weights, "before")

        # Train and evaluate the model
        model.train_first_layer(train_loader)
        model.predict_and_evaluate(train_loader, "train")
        model.predict_and_evaluate(test_loader, "test")

        # Pruning
        most_important_prune_indies, least_important_prune_indies = model.pruning(pruning_percentage=0.2)
        model.alpha_weights.data[:, least_important_prune_indies] = 0
        best_weights = model.alpha_weights.data[:, most_important_prune_indies]

        # Visualize pruned alpha weights
        pruned_alpha_weights = model.alpha_weights.detach().numpy()
        self.visualize_weights(pruned_alpha_weights, "after")

        # Train and evaluate network again with pruned weights
        model.train_first_layer(train_loader)
        model.predict_and_evaluate(train_loader, "train")
        model.predict_and_evaluate(test_loader, "test")

        # Generate new alpha weights
        model.alpha_weights = (
            nn.Parameter(torch.randn(self.first_layer_input_nodes, self.first_layer_hidden_nodes), requires_grad=True)
        )

        # Visualize new alpha weights
        new_alpha_weights = model.alpha_weights.detach().numpy()
        self.visualize_weights(new_alpha_weights, "new alpha")

        _, least_important_prune_indies = model.pruning(pruning_percentage=0.2)

        model.alpha_weights.data[:, least_important_prune_indies] = best_weights

        model.train_first_layer(train_loader)
        model.predict_and_evaluate(train_loader, "train")
        model.predict_and_evaluate(test_loader, "test")

        weights = model.alpha_weights.detach().numpy()
        self.visualize_weights(weights, "substitute")


if __name__ == "__main__":
    mpdrnn = Experiment()
    mpdrnn.main_full_sets()
