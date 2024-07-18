import colorama
import logging
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from utils.utils import measure_execution_time


class MultiPhaseDeepRandomizedNeuralNetworkBase(nn.Module):
    def __init__(self, num_data, num_features, hidden_nodes, output_nodes, activation_function, method):
        super(MultiPhaseDeepRandomizedNeuralNetworkBase, self).__init__()

        self.activation_function = (
            self.get_activation(activation_function)
        )
        self.alpha_weights = (
            nn.Parameter(torch.randn(num_features, hidden_nodes[0]), requires_grad=True)
        )
        self.beta_weights = (
            nn.Parameter(torch.zeros(hidden_nodes[0], output_nodes), requires_grad=True)
        )
        self.h1 = (
            nn.Parameter(torch.randn(num_data, hidden_nodes[0]), requires_grad=True)
        )

        self.method = method
        self.hidden_nodes = hidden_nodes

        colorama.init()

    @measure_execution_time
    def train_ith_layer(self, train_loader, hi, weights1, weights2, method, hi_prev=None):
        for train_x, train_y in tqdm(train_loader, total=len(train_loader), desc=colorama.Fore.MAGENTA + "Training"):
            if hi_prev is None:
                hi.data = self.activation_function(train_x.matmul(weights1))
            else:
                hi.data = self.activation_function(hi_prev.matmul(weights1))

            if method in ['BASE', 'EXP_ORT']:
                weights2.data = torch.linalg.pinv(hi, rcond=1e-15).matmul(train_y)
            else:
                C = 0.6
                identity_matrix = torch.eye(hi.shape[1])
                if hi.shape[0] > hi.shape[1]:
                    weights2.data = (
                        torch.linalg.pinv(hi.T.matmul(hi) + identity_matrix / C, rcond=1e-15).matmul(hi.T.matmul(train_y))
                    )
                elif hi.shape[0] < hi.shape[1]:
                    weights2.data = (
                        hi.T.matmul(
                            torch.linalg.pinv(hi.matmul(hi.T) + identity_matrix / C, rcond=1e-15).matmul(train_y)
                        )
                    )

    def train_layer(self, train_loader):
        return self.train_ith_layer(train_loader, self.h1, self.alpha_weights, self.beta_weights, self.method)

    @staticmethod
    def prune_weights(weights, pruning_percentage: float, pruning_method: str):
        abs_weights = torch.abs(weights)

        if pruning_method == "max_rank":
            ranking_matrix = abs_weights.argsort(dim=0).argsort(dim=0)
            importance_score, not_used = torch.max(ranking_matrix, dim=1)
        elif pruning_method == "sum_weight":
            importance_score = torch.sum(abs_weights, dim=1)
        else:
            raise ValueError("Pruning method must be either 'max_rank' or 'sum_weight'")

        num_neurons_to_prune = int(pruning_percentage * abs_weights.shape[0])
        least_important_prune_indices = torch.argsort(importance_score)[:num_neurons_to_prune]
        most_important_prune_indices = torch.argsort(importance_score, descending=True)[:num_neurons_to_prune]

        return most_important_prune_indices, least_important_prune_indices

    def pruning(self, pruning_percentage: float, pruning_method: str):
        return self.prune_weights(self.beta_weights, pruning_percentage, pruning_method)

    @staticmethod
    def forward(hidden_layer, weights):
        return hidden_layer.matmul(weights)

    def predict_and_evaluate(
            self, dataloader, operation: str, layer_weights=None, num_hidden_layers=None, verbose: bool = True
    ):
        for x, y in tqdm(dataloader, total=len(dataloader), desc=f"Predicting {operation} set"):
            h1 = self.activation_function(x.matmul(self.alpha_weights))
            if num_hidden_layers == 1:
                predictions = self.forward(h1, layer_weights)
            elif num_hidden_layers == 2:
                h2 = self.activation_function(h1.matmul(layer_weights[0]))
                predictions = self.forward(h2, layer_weights[1])
            elif num_hidden_layers == 3:
                h2 = self.activation_function(h1.matmul(layer_weights[0]))
                h3 = self.activation_function(h2.matmul(layer_weights[1]))
                predictions = self.forward(h3, layer_weights[2])
            else:
                raise ValueError(f"Number of hidden layers must be either 1 or 2 or 3 ")

            y_predicted_argmax = torch.argmax(predictions, dim=-1)
            y_true_argmax = torch.argmax(y, dim=-1)
            accuracy = accuracy_score(y_true_argmax, y_predicted_argmax)
            precision = precision_score(y_true_argmax, y_predicted_argmax, average='macro', zero_division=0)
            recall = recall_score(y_true_argmax, y_predicted_argmax, average='macro', zero_division=0)
            f1sore = f1_score(y_true_argmax, y_predicted_argmax, average='macro', zero_division=0)
            cm = confusion_matrix(y_true_argmax, y_predicted_argmax)

            if verbose:
                # Using getattr and setattr to dynamically set attributes
                setattr(self, f"{operation}_accuracy", accuracy)
                setattr(self, f"{operation}_precision", precision)
                setattr(self, f"{operation}_recall", recall)
                setattr(self, f"{operation}_f1sore", f1sore)
                setattr(self, f"{operation}_cm", cm)

                logging.info(f"{operation} accuracy: {accuracy:.4f}")
                logging.info(f"{operation} precision: {precision:.4f}")
                logging.info(f"{operation} recall: {recall:.4f}")
                logging.info(f"{operation} F1-score: {f1sore:.4f}")

            return [accuracy, precision, recall, f1sore, cm]

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
            num_data=base_instance.h1.size(0),
            num_features=base_instance.alpha_weights.size(0),
            hidden_nodes=base_instance.hidden_nodes,
            output_nodes=base_instance.beta_weights.size(1),
            activation_function=base_instance.activation_function.__class__.__name__,
            method=base_instance.method,
        )

        self.alpha_weights.data = base_instance.alpha_weights.data.clone()
        self.beta_weights.data = base_instance.beta_weights.data.clone()
        self.h1.data = base_instance.h1.data.clone()

        self.mu = mu
        self.sigma = sigma
        self.n_hidden_nodes = base_instance.hidden_nodes
        self.method = base_instance.method

        self.extended_beta_weights = self.create_hidden_layer(self.beta_weights)

        self.h2 = (
            nn.Parameter(torch.randn(self.n_hidden_nodes[1], self.extended_beta_weights.size(1)), requires_grad=True)
        )
        self.gamma_weights = (
            nn.Parameter(torch.randn(self.extended_beta_weights.size(1), self.beta_weights.size(1)), requires_grad=True)
        )

    def create_hidden_layer(self, weights):
        return self._create_hidden_layer(weights, self.n_hidden_nodes[1])

    def _create_hidden_layer(self, weights, n_hidden_nodes):
        noise = torch.normal(mean=self.mu, std=self.sigma, size=(weights.shape[0], weights.shape[1]))
        w_rnd_out_i = weights + noise
        hidden_layer_i_a = torch.hstack((weights, w_rnd_out_i))

        if self.method not in ["EXP_ORT", "EXP_ORT_C"]:
            w_rnd = torch.normal(mean=self.mu, std=self.sigma, size=(weights.shape[0], n_hidden_nodes))
            hidden_layer_i = torch.hstack((hidden_layer_i_a, w_rnd))
        else:
            w_rnd = torch.normal(mean=self.mu, std=self.sigma, size=(weights.shape[0], n_hidden_nodes // 2))
            q, _ = torch.linalg.qr(w_rnd)
            orthogonal_matrix = torch.mm(q, q.t())
            hidden_layer_i = torch.cat((hidden_layer_i_a, orthogonal_matrix), dim=1)

        return hidden_layer_i

    def train_layer(self, train_loader):
        super(MultiPhaseDeepRandomizedNeuralNetworkSubsequent, self).train_ith_layer(
            train_loader=train_loader,
            hi=self.h2,
            weights1=self.extended_beta_weights,
            weights2=self.gamma_weights,
            method=self.method,
            hi_prev=self.h1
        )

    def predict_and_evaluate(
            self, dataloader, operation: str, layer_weights=None, num_hidden_layers=None, verbose: bool = True
    ):
        super(MultiPhaseDeepRandomizedNeuralNetworkSubsequent, self).predict_and_evaluate(
            dataloader, operation, layer_weights, num_hidden_layers, verbose
        )

    def pruning(self, pruning_percentage: float, pruning_method: str):
        return self.prune_weights(self.gamma_weights, pruning_percentage, pruning_method)


class MultiPhaseDeepRandomizedNeuralNetworkFinal(MultiPhaseDeepRandomizedNeuralNetworkSubsequent):
    def __init__(self, base_instance, mu, sigma):
        super(MultiPhaseDeepRandomizedNeuralNetworkFinal, self).__init__(
            base_instance,
            mu,
            sigma
        )

        self.alpha_weights.data = base_instance.alpha_weights.data.clone()
        self.beta_weights.data = base_instance.beta_weights.data.clone()
        self.extended_beta_weights.data = base_instance.extended_beta_weights.data.clone()
        self.gamma_weights.data = base_instance.gamma_weights.data.clone()

        self.h1.data = base_instance.h1.data.clone()
        self.h2.data = base_instance.h2.data.clone()

        self.mu = mu
        self.sigma = sigma

        self.n_hidden_nodes = base_instance.hidden_nodes
        self.method = base_instance.method

        self.extended_gamma_weights = self.create_hidden_layer(self.gamma_weights)

        self.h3 = nn.Parameter(torch.randn(self.h2.size()), requires_grad=True)
        self.delta_weights = nn.Parameter(torch.randn(self.h2.size(1), self.beta_weights.size(1)), requires_grad=True)

    def create_hidden_layer(self, weights):
        return self._create_hidden_layer(weights, self.n_hidden_nodes[2])

    def train_layer(self, train_loader):
        return super(MultiPhaseDeepRandomizedNeuralNetworkSubsequent, self).train_ith_layer(
            train_loader=train_loader,
            hi=self.h3,
            weights1=self.extended_gamma_weights,
            weights2=self.delta_weights,
            method=self.method,
            hi_prev=self.h2
        )

    def predict_and_evaluate(
            self, dataloader, operation: str, layer_weights=None, num_hidden_layers=None, verbose: bool = True
    ):
        return super(MultiPhaseDeepRandomizedNeuralNetworkSubsequent, self).predict_and_evaluate(
            dataloader, operation, layer_weights, num_hidden_layers, verbose
        )
