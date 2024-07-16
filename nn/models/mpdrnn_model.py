import colorama
import logging
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score
from tqdm import tqdm


class MultiPhaseDeepRandomizedNeuralNetworkBase(nn.Module):
    def __init__(self, num_data, num_features, hidden_nodes, output_nodes, activation_function):
        super(MultiPhaseDeepRandomizedNeuralNetworkBase, self).__init__()

        self.activation_function = self.get_activation(activation_function)
        self.alpha_weights = nn.Parameter(torch.randn(num_features, hidden_nodes), requires_grad=True)
        self.beta_weights = nn.Parameter(torch.zeros(hidden_nodes, output_nodes), requires_grad=True)
        self.h1 = nn.Parameter(torch.randn(num_data, hidden_nodes), requires_grad=True)

        colorama.init()

    def forward(self, hidden_layer_1):
        return hidden_layer_1 @ self.beta_weights

    def train_first_layer(self, train_loader):
        for train_x, train_y in tqdm(train_loader, total=len(train_loader), desc=colorama.Fore.MAGENTA + "Training"):
            self.h1.data = self.activation_function(train_x @ self.alpha_weights)
            self.beta_weights.data = torch.linalg.pinv(self.h1).matmul(train_y)

    def pruning(self, pruning_percentage: float, pruning_method: str):
        abs_beta_weights = torch.abs(self.beta_weights)

        if pruning_method == "max_rank":
            ranking_matrix = abs_beta_weights.argsort(dim=0).argsort(dim=0)
            importance_score, not_used = torch.max(ranking_matrix, dim=1)
        elif pruning_method == "sum_weight":
            importance_score = torch.sum(abs_beta_weights, dim=1)
        else:
            raise ValueError("Pruning method must be either 'max_rank' or 'sum_weight'")

        num_neurons_to_prune = int(pruning_percentage * abs_beta_weights.shape[0])
        least_important_prune_indices = torch.argsort(importance_score)[:num_neurons_to_prune]
        most_important_prune_indices = torch.argsort(importance_score, descending=True)[:num_neurons_to_prune]

        return most_important_prune_indices, least_important_prune_indices

    def predict_and_evaluate(self, dataloader, operation: str, verbose: bool = True):
        for x, y in tqdm(dataloader, total=len(dataloader), desc=f"Predicting {operation} set"):
            h1 = self.activation_function(x @ self.alpha_weights)
            predictions = self.forward(h1)

            y_predicted_argmax = torch.argmax(predictions, dim=-1)
            y_true_argmax = torch.argmax(y, dim=-1)
            accuracy = accuracy_score(y_true_argmax, y_predicted_argmax)

            if verbose:
                logging.info(f"Accuracy of {operation} set: {accuracy:.4f}")

            return accuracy

    @staticmethod
    def get_activation(activation):
        activation_map = {
            "Sigmoid": nn.Sigmoid(),
            "Identity": nn.Identity(),
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU(negative_slope=0.2),
            "Tanh": nn.Tanh(),
        }

        return activation_map[activation]


class MultiPhaseDeepRandomizedNeuralNetworkSubsequent(MultiPhaseDeepRandomizedNeuralNetworkBase):
    def __init__(self, base_instance, mu, sigma):
        super(MultiPhaseDeepRandomizedNeuralNetworkSubsequent, self).__init__(
            base_instance.h1.size(0),                               # num_data
            base_instance.alpha_weights.size(0),                    # num_features
            base_instance.alpha_weights.size(1),                    # hidden_nodes
            base_instance.beta_weights.size(1),                     # output_nodes
            base_instance.activation_function.__class__.__name__    # activation_function
        )

        self.alpha_weights.data = base_instance.alpha_weights.data.clone()
        self.beta_weights.data = base_instance.beta_weights.data.clone()
        self.h1.data = base_instance.h1.data.clone()

        self.mu = mu
        self.sigma = sigma
        self.n_hidden_nodes = base_instance.alpha_weights.size(1)

        self.extended_beta_weights = self.create_hidden_layer_2()

        self.h2 = (
            nn.Parameter(torch.randn(self.n_hidden_nodes, self.extended_beta_weights.size(1)), requires_grad=True)
        )
        self.gamma_weights = (
            nn.Parameter(torch.randn(self.extended_beta_weights.size(1), self.beta_weights.size(1)), requires_grad=True)
        )

    def create_hidden_layer_2(self):
        noise = torch.normal(mean=self.mu, std=self.sigma, size=(self.beta_weights.shape[0],
                                                                 self.beta_weights.shape[1]))
        w_rnd_out_i = self.beta_weights + noise
        hidden_layer_i_a = torch.hstack((self.beta_weights, w_rnd_out_i))

        w_rnd = torch.normal(mean=self.mu, std=self.sigma, size=(self.beta_weights.shape[0],
                                                                 self.n_hidden_nodes))
        hidden_layer_i = torch.hstack((hidden_layer_i_a, w_rnd))

        return hidden_layer_i

    def forward(self, hidden_layer_2):
        return hidden_layer_2 @ self.gamma_weights

    def train_second_layer(self, train_loader):
        for _, train_y in tqdm(train_loader, total=len(train_loader), desc="Training"):
            self.h2.data = self.activation_function(self.h1 @ self.extended_beta_weights)
            self.gamma_weights.data = torch.linalg.pinv(self.h2).matmul(train_y)

    def predict_and_evaluate(self, dataloader, operation: str, verbose: bool = True):
        for x, y in tqdm(dataloader, total=len(dataloader), desc=f"Predicting {operation} set"):
            h1 = self.activation_function(x @ self.alpha_weights)
            h2 = self.activation_function(h1 @ self.extended_beta_weights)
            predictions = self.forward(h2)

            y_predicted_argmax = torch.argmax(predictions, dim=-1)
            y_true_argmax = torch.argmax(y, dim=-1)
            accuracy = accuracy_score(y_true_argmax, y_predicted_argmax)

            if verbose:
                logging.info(f"Accuracy of {operation} set: {accuracy:.4f}")

            return accuracy

    def pruning(self, pruning_percentage: float, pruning_method: str):
        abs_gamma_weights = torch.abs(self.gamma_weights)

        if pruning_method == "max_rank":
            ranking_matrix = abs_gamma_weights.argsort(dim=0).argsort(dim=0)
            importance_score, not_used = torch.max(ranking_matrix, dim=1)
        elif pruning_method == "sum_weight":
            importance_score = torch.sum(abs_gamma_weights, dim=1)
        else:
            raise ValueError("Pruning method must be either 'max_rank' or 'sum_weight'")

        num_neurons_to_prune = int(pruning_percentage * abs_gamma_weights.shape[0])
        least_important_prune_indices = torch.argsort(importance_score)[:num_neurons_to_prune]
        most_important_prune_indices = torch.argsort(importance_score, descending=True)[:num_neurons_to_prune]

        return most_important_prune_indices, least_important_prune_indices
