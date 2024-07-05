import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from sklearn.metrics import accuracy_score
from tqdm import tqdm
from nn.dataloader.npz_dataloader import NpzDataset
from torch.utils.data import DataLoader, Subset

from config.config import MPDRNNConfig
from config.dataset_config import elm_general_dataset_configs
from utils.utils import display_dataset_info, setup_logger


class Model(nn.Module):
    def __init__(self, num_data, num_features, hidden_nodes, output_nodes, activation_function):
        super(Model, self).__init__()
        self.activation_function = self.get_activation(activation_function)
        self.alpha_weights = nn.Parameter(torch.randn(num_features, hidden_nodes), requires_grad=True)
        self.beta_weights = nn.Parameter(torch.zeros(hidden_nodes, output_nodes), requires_grad=True)
        self.h1 = nn.Parameter(torch.randn(num_data, hidden_nodes), requires_grad=True)

    def forward(self, hidden_layer_1):
        return hidden_layer_1 @ self.beta_weights

    def train_first_layer(self, train_loader):
        for train_x, train_y in tqdm(train_loader, total=len(train_loader), desc="Training"):
            self.h1.data = self.activation_function(train_x @ self.alpha_weights)
            self.beta_weights.data = torch.pinverse(self.h1).matmul(train_y)

    def predict_and_evaluate(self, dataloader):
        for _, y in tqdm(dataloader, total=len(dataloader), desc=f"Predicting"):
            predictions = self.forward(self.h1)

            y_predicted_argmax = torch.argmax(predictions, dim=-1)
            y_true_argmax = torch.argmax(y, dim=-1)
            accuracy = accuracy_score(y_true_argmax, y_predicted_argmax)

        print(f"Accuracy: {accuracy}")

    @staticmethod
    def get_activation(activation):
        activation_map = {
            "sigmoid": nn.Sigmoid(),
            "identity": nn.Identity(),
            "ReLU": nn.ReLU(),
            "leaky_ReLU": nn.LeakyReLU(negative_slope=0.2),
            "tanh": nn.Tanh(),
        }

        return activation_map[activation]


class MultiPhaseDeepRandomizedNeuralNetwork:
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

    @staticmethod
    def create_subset(dataset, percentage):
        total_samples = len(dataset)
        subset_size = int(total_samples * percentage)
        indices = list(range(total_samples))

        selected_indices = indices[:subset_size]
        subset = Subset(dataset, selected_indices)

        return subset

    def create_datasets(self, file_path, subset_percentage):
        full_train_dataset = NpzDataset(file_path, operation="train")
        test_dataset = NpzDataset(file_path, operation="test")

        train_dataset = self.create_subset(full_train_dataset, subset_percentage)

        train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

        return train_loader, test_loader

    @staticmethod
    def replace_zeros_with_random(module, param_name):
        param = getattr(module, param_name)
        mask = param == 0
        random_values = torch.FloatTensor(np.random.uniform(-1, 1, mask.sum().item()))
        param.data[mask] = random_values

    @staticmethod
    def check_for_zeros(parameters):
        for module, param_name in parameters:
            param = getattr(module, param_name)
            if torch.any(param == 0):
                print(f"Parameter {param_name} in module {module} contains zeros.")
            else:
                print(f"Parameter {param_name} in module {module} does not contain any zeros.")

    def main(self):
        model = Model(num_data=self.first_layer_num_data,
                      num_features=self.first_layer_input_nodes,
                      hidden_nodes=self.first_layer_hidden_nodes,
                      output_nodes=self.first_layer_output_nodes,
                      activation_function="leaky_ReLU")

        file_path = elm_general_dataset_configs(self.cfg).get("cached_dataset_file")
        train_loader, test_loader = self.create_datasets(file_path, self.cfg.subset_percentage)

        model.train_first_layer(train_loader)
        model.predict_and_evaluate(train_loader)

        # Prune the parameters by magnitude
        prune.l1_unstructured(model, name="alpha_weights", amount=0.2)

        # Remove the pruning re-parameterization to make it permanent
        prune.remove(model, name="alpha_weights")

        self.replace_zeros_with_random(model, "alpha_weights")

        model.train_first_layer(train_loader)
        model.predict_and_evaluate(train_loader)


if __name__ == "__main__":
    mpdrnn = MultiPhaseDeepRandomizedNeuralNetwork()
    mpdrnn.main()
