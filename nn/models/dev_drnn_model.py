import colorama
import logging
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from typing import Any

from utils.utils import measure_execution_time


class DevDeepRandomizedNeuralNetworkFirstLayer(nn.Module):
    def __init__(
        self,
        num_data: int,
        num_features: int,
        hidden_nodes: list[int],
        output_nodes: int,
        activation_function: str,
        rcond: float,
        penalty_term: float = None,
        train_loader=None,
    ):
        """
        Initialize the DevDeepRandomizedNeuralNetworkFirstLayer class.

        Args:
            num_data (int): Number of data samples.
            num_features (int): Number of input features.
            hidden_nodes (list[int]): List containing number of hidden nodes for each hidden layer.
            output_nodes (int): Number of output nodes.
            activation_function (str): Activation function to be used in the network.
            penalty_term (float): Penalty term to be used in the network.
        """

        super(DevDeepRandomizedNeuralNetworkFirstLayer, self).__init__()

        self.activation_function = self.get_activation(activation_function)
        self.train_loader = train_loader

        self.alpha_weights = nn.Parameter(torch.randn(num_features, hidden_nodes[0]), requires_grad=True)
        self.beta_weights = nn.Parameter(torch.zeros(hidden_nodes[0], output_nodes), requires_grad=True)
        self.h1 = nn.Parameter(torch.zeros(num_data, hidden_nodes[0]), requires_grad=True)

        self.predictions = None

        self.hidden_nodes = hidden_nodes
        self.rcond = rcond
        self.penalty_term = penalty_term

        colorama.init()

    @measure_execution_time
    def train_ith_layer(
        self,
        hi: torch.Tensor,
        weights1: nn.Parameter,
        weights2: nn.Parameter,
        hi_prev: torch.Tensor = None,
    ) -> None:
        """
        Train the i-th layer of the network.

        Args:
            hi (torch.Tensor): Current hidden layer tensor.
            weights1 (nn.Parameter): Weights for the current layer.
            weights2 (nn.Parameter): Weights for the next layer.
            hi_prev (torch.Tensor, optional): Previous hidden layer tensor. Defaults to None.
        Returns:
            None
        """

        for train_x, train_y in tqdm(
            self.train_loader, total=len(self.train_loader), desc=colorama.Fore.MAGENTA + "Training"
        ):
            if hi_prev is None:
                hi.data = self.activation_function(train_x @ weights1)
            else:
                hi.data = self.activation_function(hi_prev @ weights1)

            identity_matrix = torch.eye(hi.shape[1], device=hi.device)
            if hi.shape[0] > hi.shape[1]:
                pseudo_inv_input = hi.T @ hi + identity_matrix / self.penalty_term
                if self.rcond is not None:
                    weights2.data = torch.linalg.pinv(pseudo_inv_input, rcond=self.rcond) @ (hi.T @ train_y)
                else:
                    weights2.data = torch.linalg.pinv(pseudo_inv_input) @ (hi.T @ train_y)
            else:
                pseudo_inv_input = hi @ hi.T + identity_matrix / self.penalty_term
                if self.rcond is not None:
                    weights2.data = hi.T @ torch.linalg.pinv(pseudo_inv_input, rcond=self.rcond) @ train_y
                else:
                    weights2.data = hi.T @ torch.linalg.pinv(pseudo_inv_input) @ train_y

    def train_layer(self):
        """
        Train the first layer of the network.

        Returns:
            None
        """

        return self.train_ith_layer(hi=self.h1, weights1=self.alpha_weights, weights2=self.beta_weights)

    @staticmethod
    def forward(hidden_layer: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass by multiplying the hidden layer with weights.

        Args:
            hidden_layer (torch.Tensor): The hidden layer tensor.
            weights (torch.Tensor): The weight tensor.

        Returns:
            torch.Tensor: The output tensor after applying weights to hidden layer.

        """

        return hidden_layer.matmul(weights)

    def predict_and_evaluate(
        self,
        dataloader,
        operation: str,
        layer_weights: torch.Tensor = None,
        num_hidden_layers: int = None,
        verbose: bool = True,
    ) -> list[float | Any] | None:
        """
        Predict and evaluate the performance of the network and compute class error vectors.
        Assumes the dataloader contains only one batch (all data).

        Returns:
            list: [accuracy, precision, recall, F1-score, confusion matrix, class_error_vectors]
        """
        x, y = next(iter(dataloader))

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
            raise ValueError("Number of hidden layers must be 1, 2, or 3")

        if operation == "train":
            self.predictions = predictions
        y_pred_argmax = torch.argmax(predictions, dim=-1)
        y_true_argmax = torch.argmax(y, dim=-1)

        accuracy = accuracy_score(y_true_argmax, y_pred_argmax)
        precision = precision_score(y_true_argmax, y_pred_argmax, average="macro", zero_division=0)
        recall = recall_score(y_true_argmax, y_pred_argmax, average="macro", zero_division=0)
        f1sore = f1_score(y_true_argmax, y_pred_argmax, average="macro", zero_division=0)
        cm = confusion_matrix(y_true_argmax, y_pred_argmax)

        if verbose:
            setattr(self, f"{operation}_accuracy", accuracy)
            setattr(self, f"{operation}_precision", precision)
            setattr(self, f"{operation}_recall", recall)
            setattr(self, f"{operation}_f1sore", f1sore)
            setattr(self, f"{operation}_cm", cm)

            logging.info(f"{operation} accuracy: {accuracy:.4f}")
            logging.info(f"{operation} precision: {precision:.4f}")
            logging.info(f"{operation} recall: {recall:.4f}")
            logging.info(f"{operation} F1-score: {f1sore:.4f}")

        return [accuracy, precision, recall, f1sore, cm, predictions]

    def compute_class_mean(self, hidden_layer):
        _, y = next(iter(self.train_loader))
        y_true = torch.argmax(y, dim=1)

        classes = torch.unique(y_true)

        # mean vector per class (vector)
        class_mean_vectors = {}
        for c in classes:
            mask = y_true == c
            class_mean_vectors[int(c)] = hidden_layer[mask].mean(dim=0)

        return class_mean_vectors

    def compute_class_error_vectors(self) -> dict:
        """
        Compute class-wise average error vectors based on model predictions.

        For each class, this method computes the mean prediction error vector
        (prediction minus ground truth) over all samples belonging to that class.

        Returns:
            dict:
                A dictionary mapping class labels (int) to their corresponding
                mean error vectors (torch.Tensor of shape [num_classes]).
        """

        _, y = next(iter(self.train_loader))

        errors = self.predictions - y
        y_true = torch.argmax(y, dim=-1)

        class_error_vectors = {}
        for class_name in torch.unique(y_true, sorted=True):
            mask = y_true == class_name
            class_error_vectors[int(class_name)] = errors[mask].mean(dim=0)

        return class_error_vectors

    @staticmethod
    def compute_class_distance_matrix(class_mean_vectors: dict):
        classes = sorted(class_mean_vectors.keys())

        mean_matrix = torch.stack([class_mean_vectors[c] for c in classes])
        distance_matrix = mean_matrix.unsqueeze(1) - mean_matrix.unsqueeze(0)

        return distance_matrix

    @staticmethod
    def compute_class_error_distance_matrix(class_error_vectors: dict):
        """
        Compute a pairwise distance matrix between classes based on their error vectors.

        The distance between two classes is defined as the L2 (Euclidean) norm
        of the difference between their corresponding class error vectors.

        Args:
            class_error_vectors (dict):
                A dictionary mapping class labels to error vectors.

        Returns:
            torch.Tensor:
                A square matrix of shape (C, C), where C is the number of classes,
                containing pairwise class distances.
        """

        classes = sorted(class_error_vectors.keys())
        error_matrix = torch.stack([class_error_vectors[c] for c in classes])

        diff = error_matrix.unsqueeze(1) - error_matrix.unsqueeze(0)
        distance_matrix = torch.norm(diff, dim=-1, p=2)

        return distance_matrix

    @staticmethod
    def get_sorted_class_pairs(distance_matrix: torch.Tensor):
        """
        Extract and sort all unique class pairs based on their distance.

        Only the upper triangular part of the distance matrix is considered to avoid duplicate pairs.
        Class pairs are sorted in ascending order of distance, where smaller distances indicate more confusable classes.

        Args:
            distance_matrix (torch.Tensor):
                A square matrix of pairwise class distances.

        Returns:
            list of tuples:
                Each tuple has the form (class_1, class_2, distance), sorted
                by increasing distance.
        """

        C = distance_matrix.shape[0]

        triu_indices = torch.triu_indices(C, C, offset=1)
        distances = distance_matrix[triu_indices[0], triu_indices[1]]

        sorted_idx = torch.argsort(distances, descending=False)

        pairs = [
            (triu_indices[0][idx].item(), triu_indices[1][idx].item(), distances[idx].item()) for idx in sorted_idx
        ]

        return pairs

    def _allocate_neurons_per_class_pair(self, hidden_layer, total_neurons: int, eps: float = 1e-8):
        """
        Allocate neurons to each class pair inversely proportional to their distance.

        Each class pair receives at least one neuron. The remaining neurons are
        distributed proportionally to the inverse of the class distance, such that
        more neurons are assigned to more confusable (closer) class pairs.

        Args:
            total_neurons (int):
                Total number of neurons to allocate.
            eps (float, optional):
                Small constant to avoid division by zero. Default is 1e-8.

        Returns:
            dict:
                A dictionary mapping class pairs (c1, c2) to allocated neuron counts.

        Raises:
            ValueError:
                If total_neurons is smaller than the number of class pairs.
        """

        class_mean_vector = self.compute_class_mean(hidden_layer)
        distance_matrix = self.compute_class_distance_matrix(class_mean_vectors=class_mean_vector)

        class_error_vectors = self.compute_class_error_vectors()
        distance_error_matrix = self.compute_class_error_distance_matrix(class_error_vectors)
        d_min = torch.min(distance_error_matrix)
        d_max = torch.max(distance_error_matrix)
        distance_error_matrix = (distance_error_matrix - d_min) / (d_max - d_min)

        sorted_class_pairs = self.get_sorted_class_pairs(distance_error_matrix)

        unique_class_pair = len(sorted_class_pairs)
        if total_neurons < unique_class_pair:
            raise ValueError("Total neurons must be >= number of class pairs")

        allocation = {(c1, c2): 1 for (c1, c2, _) in sorted_class_pairs}
        remaining = total_neurons - unique_class_pair

        distances = torch.tensor([d for _, _, d in sorted_class_pairs])
        # Inverse distance -> smaller distance greater weight
        inverse_distance = 1.0 / (distances + eps)
        # Normalization
        norm_inverse_distance = inverse_distance / inverse_distance.sum()
        # sort number of inverse_distance to each classes
        extra = torch.floor(norm_inverse_distance * remaining).int()

        for (c1, c2, _), n in zip(sorted_class_pairs, extra):
            allocation[(c1, c2)] += int(n)

        diff = total_neurons - sum(allocation.values())
        if diff > 0:
            for i in range(diff):
                c1, c2, _ = sorted_class_pairs[i]
                allocation[(c1, c2)] += 1
        elif diff < 0:
            for i in range(-diff):
                c1, c2, _ = sorted_class_pairs[-(i + 1)]
                allocation[(c1, c2)] -= 1

        return allocation, distance_matrix

    def allocate_neurons_per_class_pair(self, hidden_layer, total_neurons: int, eps: float = 1e-8):
        """
        Allocate neurons to class pairs based on their relative difficulty.

        This is a public wrapper around the internal allocation method.

        Args:
            hidden_layer:

            total_neurons (int):
                Total number of neurons to be allocated across all class pairs.
            eps (float, optional):
                Small constant added for numerical stability. Default is 1e-8.

        Returns:
            dict:
                A dictionary mapping class pairs (c1, c2) to the number of
                allocated neurons.
        """

        return self._allocate_neurons_per_class_pair(hidden_layer, total_neurons, eps)

    @staticmethod
    def get_activation(activation: str) -> nn.Module:
        """
        Get the activation function module.

        Args:
            activation (str): Name of the activation function.

        Returns:
            nn.Module: Corresponding activation function module.
        """

        activation_map = {
            "Sigmoid": nn.Sigmoid(),
            "Identity": nn.Identity(),
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU(negative_slope=0.2),
            "Tanh": nn.Tanh(),
        }

        return activation_map[activation]


class DevDeepRandomizedNeuralNetworkSecondLayer(DevDeepRandomizedNeuralNetworkFirstLayer):
    def __init__(self, first_layer_instance: DevDeepRandomizedNeuralNetworkFirstLayer, mu: float, sigma: float):
        """
        Initialize the DevDeepRandomizedNeuralNetworkSecondLayer class.

        Args:
            first_layer_instance (DevDeepRandomizedNeuralNetworkFirstLayer): An instance of the base class.
            mu (float): Mean for normal distribution to create noise.
            sigma (float): Standard deviation for normal distribution to create noise.
        """

        super(DevDeepRandomizedNeuralNetworkSecondLayer, self).__init__(
            num_data=first_layer_instance.h1.size(0),
            num_features=first_layer_instance.alpha_weights.size(0),
            hidden_nodes=first_layer_instance.hidden_nodes,
            output_nodes=first_layer_instance.beta_weights.size(1),
            activation_function=first_layer_instance.activation_function.__class__.__name__,
            rcond=first_layer_instance.rcond,
            penalty_term=first_layer_instance.penalty_term,
            train_loader=first_layer_instance.train_loader,
        )

        self.alpha_weights.data = first_layer_instance.alpha_weights.data.clone()
        self.beta_weights.data = first_layer_instance.beta_weights.data.clone()
        self.h1.data = first_layer_instance.h1.data.clone()
        self.n_hidden_nodes = first_layer_instance.hidden_nodes
        self.allocation, self.distance_mtx = (
            first_layer_instance.allocate_neurons_per_class_pair(
                hidden_layer=self.h1.data,
                total_neurons=self.n_hidden_nodes[1]
            )
        )

        self.mu = mu
        self.sigma = sigma
        self.rcond = first_layer_instance.rcond

        self.extended_beta_weights = self.create_hidden_layer(self.beta_weights)

        self.h2 = nn.Parameter(
            torch.zeros(self.n_hidden_nodes[1], self.extended_beta_weights.size(1)), requires_grad=True
        )
        self.gamma_weights = nn.Parameter(
            torch.zeros(self.extended_beta_weights.size(1), self.beta_weights.size(1)), requires_grad=True
        )

    def create_hidden_layer(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Create a hidden layer with added noise based on the given weights.

        Args:
            weights (torch.Tensor): The weight tensor to create the hidden layer from.

        Returns:
            torch.Tensor: The created hidden layer with added noise.
        """

        return self._create_hidden_layer(weights)

    def _create_hidden_layer(self, weights):
        dimension, _ = weights.shape

        noise = torch.normal(mean=self.mu, std=self.sigma, size=weights.shape)
        w_rnd_out_i = weights + noise
        hidden_layer_i_a = torch.hstack((weights, w_rnd_out_i))

        new_columns = []

        for (class_1, class_2), neuron_per_class in self.allocation.items():
            base_vec = self.distance_mtx[class_1, class_2]
            base_vec = base_vec / (base_vec.norm() + 1e-8)

            new_columns.append(base_vec.view(dimension, 1))

            for _ in range(max(neuron_per_class - 1, 0)):
                noise = torch.normal(mean=0.0, std=self.sigma, size=(dimension,))
                new_vec = base_vec + noise
                new_columns.append(new_vec.view(dimension, 1))

        if new_columns:
            hidden_new = torch.cat(new_columns, dim=1)
            hidden_layer_i = torch.cat((hidden_layer_i_a, hidden_new), dim=1)
        else:
            hidden_layer_i = hidden_layer_i_a

        return hidden_layer_i

    def train_layer(self):
        """
        Train the next layer of the network.

        Returns:
            None
        """

        super(DevDeepRandomizedNeuralNetworkSecondLayer, self).train_ith_layer(
            hi=self.h2,
            weights1=self.extended_beta_weights,
            weights2=self.gamma_weights,
            hi_prev=self.h1,
        )

    def predict_and_evaluate(
        self, dataloader, operation: str, layer_weights=None, num_hidden_layers: int = None, verbose: bool = True
    ):
        """
        Predict and evaluate the performance of the second layer.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
            operation (str): Operation being performed ('train' or 'test').
            layer_weights (torch.Tensor, optional): Weights for the layers. Defaults to None.
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to None.
            verbose (bool, optional): Whether to print detailed logs. Defaults to True.

        Returns:
            list: List containing accuracy, precision, recall, F1-score, and confusion matrix.
        """

        return super(DevDeepRandomizedNeuralNetworkSecondLayer, self).predict_and_evaluate(
            dataloader, operation, layer_weights, num_hidden_layers, verbose
        )

    def allocate_neurons_per_class_pair(self, hidden_layer, total_neurons: int, eps: float = 1e-8):
        return super(DevDeepRandomizedNeuralNetworkSecondLayer, self).allocate_neurons_per_class_pair(
            hidden_layer,
            total_neurons,
            eps
        )


class DevDeepRandomizedNeuralNetworkThirdLayer(DevDeepRandomizedNeuralNetworkSecondLayer):
    def __init__(self, second_layer_instance: DevDeepRandomizedNeuralNetworkSecondLayer, mu: float, sigma: float):
        """
        Initialize the DevDeepRandomizedNeuralNetworkThirdLayer class.

        Args:
            second_layer_instance (DevDeepRandomizedNeuralNetworkSecondLayer): An instance of the second layer class.
            mu (float): Mean for normal distribution to create noise.
            sigma (float): Standard deviation for normal distribution to create noise.
        """

        super(DevDeepRandomizedNeuralNetworkThirdLayer, self).__init__(second_layer_instance, mu, sigma)

        self.alpha_weights.data = second_layer_instance.alpha_weights.data.clone()
        self.beta_weights.data = second_layer_instance.beta_weights.data.clone()
        self.extended_beta_weights.data = second_layer_instance.extended_beta_weights.data.clone()
        self.gamma_weights.data = second_layer_instance.gamma_weights.data.clone()

        self.h1.data = second_layer_instance.h1.data.clone()
        self.h2.data = second_layer_instance.h2.data.clone()

        self.mu = mu
        self.sigma = sigma
        self.rcond = second_layer_instance.rcond

        self.n_hidden_nodes = second_layer_instance.hidden_nodes

        self.allocation, self.distance_mtx = second_layer_instance.allocate_neurons_per_class_pair(
            hidden_layer=self.h2.data,
            total_neurons=self.n_hidden_nodes[2]
        )
        self.extended_gamma_weights = self.create_hidden_layer(self.gamma_weights)

        self.h3 = nn.Parameter(torch.randn(self.h2.size()), requires_grad=True)
        self.delta_weights = nn.Parameter(torch.randn(self.h2.size(1), self.beta_weights.size(1)), requires_grad=True)

    def create_hidden_layer(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Create a hidden layer with added noise based on the given weights.

        Args:
            weights (torch.Tensor): The weight tensor to create the hidden layer from.

        Returns:
            torch.Tensor: The created hidden layer with added noise.
        """

        return self._create_hidden_layer(weights)

    def train_layer(self):
        """
        Train the next layer of the network.
        """

        return super(DevDeepRandomizedNeuralNetworkSecondLayer, self).train_ith_layer(
            hi=self.h3,
            weights1=self.extended_gamma_weights,
            weights2=self.delta_weights,
            hi_prev=self.h2,
        )

    def predict_and_evaluate(
        self,
        dataloader,
        operation: str,
        layer_weights: torch.Tensor = None,
        num_hidden_layers: int = None,
        verbose: bool = True,
    ) -> list:
        """
        Predict and evaluate the performance of the final network layer.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
            operation (str): Operation being performed ('train' or 'test').
            layer_weights (torch.Tensor, optional): Weights for the layers. Defaults to None.
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to None.
            verbose (bool, optional): Whether to print detailed logs. Defaults to True.

        Returns:
            list: List containing accuracy, precision, recall, F1-score, and confusion matrix.
        """

        return super(DevDeepRandomizedNeuralNetworkSecondLayer, self).predict_and_evaluate(
            dataloader, operation, layer_weights, num_hidden_layers, verbose
        )
