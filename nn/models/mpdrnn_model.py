import colorama
import logging
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from utils.utils import measure_execution_time


class MultiPhaseDeepRandomizedNeuralNetworkBase(nn.Module):
    def __init__(self, num_data: int, num_features: int, hidden_nodes: list[int], output_nodes: int,
                 activation_function: str, method: str, penalty_term: float = None):
        """
       Initialize the MultiPhaseDeepRandomizedNeuralNetworkBase class.

       Args:
           num_data (int): Number of data samples.
           num_features (int): Number of input features.
           hidden_nodes (list[int]): List containing number of hidden nodes for each hidden layer.
           output_nodes (int): Number of output nodes.
           activation_function (str): Activation function to be used in the network.
           method (str): Method to be used for training the network.
           penalty_term (float): Penalty term to be used in the network.
       """

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
            nn.Parameter(torch.zeros(num_data, hidden_nodes[0]), requires_grad=True)
        )

        self.method = method
        self.hidden_nodes = hidden_nodes

        if self.method == "EXP_ORT_C" and penalty_term is not None:
            self.penalty_term = penalty_term

        colorama.init()

    @measure_execution_time
    def train_ith_layer(self, train_loader, hi: torch.Tensor, weights1: torch.Tensor, weights2: torch.Tensor,
                        method: str, hi_prev: torch.Tensor = None):
        """
        Train the i-th layer of the network.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            hi (torch.Tensor): Current hidden layer tensor.
            weights1 (nn.Parameter): Weights for the current layer.
            weights2 (nn.Parameter): Weights for the next layer.
            method (str): Method to be used for training the network.
            hi_prev (torch.Tensor, optional): Previous hidden layer tensor. Defaults to None.

        Returns:
            None
        """

        for train_x, train_y in tqdm(train_loader, total=len(train_loader), desc=colorama.Fore.MAGENTA + "Training"):
            if hi_prev is None:
                hi.data = self.activation_function(train_x.matmul(weights1))
            else:
                hi.data = self.activation_function(hi_prev.matmul(weights1))

            if method in ['BASE', 'EXP_ORT']:
                weights2.data = torch.linalg.pinv(hi, rcond=1e-15).matmul(train_y)
            else:
                identity_matrix = torch.eye(hi.shape[1])
                if hi.shape[0] > hi.shape[1]:
                    weights2.data = (
                        torch.linalg.pinv(hi.T.matmul(hi) + identity_matrix / self.penalty_term, rcond=1e-15).matmul(hi.T.matmul(train_y))
                    )
                else:
                    weights2.data = (
                        hi.T.matmul(
                            torch.linalg.pinv(hi.matmul(hi.T) + identity_matrix / self.penalty_term, rcond=1e-15).matmul(train_y)
                        )
                    )

    def train_layer(self, train_loader):
        """
        Train the first layer of the network.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.

        Returns:
            None
        """

        return self.train_ith_layer(train_loader, self.h1, self.alpha_weights, self.beta_weights, self.method)

    @staticmethod
    def prune_weights(weights: torch.Tensor, pruning_percentage: float, pruning_method: str) \
            -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prune weights based on a specified method and pruning percentage.

        Args:
            weights (torch.Tensor): Weight tensor to be pruned.
            pruning_percentage (float): Percentage of weights to be pruned.
            pruning_method (str): Method to be used for pruning ('max_rank' or 'sum_weight').

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Indices of most important and least important weights.
        """

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

    def pruning(self, pruning_percentage: float, pruning_method: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prune the beta weights of the network.

        Args:
            pruning_percentage (float): Percentage of weights to be pruned.
            pruning_method (str): Method to be used for pruning ('max_rank' or 'sum_weight').

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Indices of most important and least important weights.
        """

        return self.prune_weights(self.beta_weights, pruning_percentage, pruning_method)

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
            self, dataloader, operation: str, layer_weights: torch.Tensor = None, num_hidden_layers: int = None,
            verbose: bool = True) -> list:
        """
        Predict and evaluate the performance of the network.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
            operation (str): Operation being performed ('train' or 'test').
            layer_weights (torch.Tensor, optional): Weights for the layers. Defaults to None.
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to None.
            verbose (bool, optional): Whether to print detailed logs. Defaults to True.

        Returns:
            list: List containing accuracy, precision, recall, F1-score, and confusion matrix.
        """

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


class MultiPhaseDeepRandomizedNeuralNetworkSubsequent(MultiPhaseDeepRandomizedNeuralNetworkBase):
    def __init__(self, base_instance: MultiPhaseDeepRandomizedNeuralNetworkBase, mu: float, sigma: float):
        """
        Initialize the MultiPhaseDeepRandomizedNeuralNetworkSubsequent class.

        Args:
            base_instance (MultiPhaseDeepRandomizedNeuralNetworkBase): An instance of the base class.
            mu (float): Mean for normal distribution to create noise.
            sigma (float): Standard deviation for normal distribution to create noise.
        """

        super(MultiPhaseDeepRandomizedNeuralNetworkSubsequent, self).__init__(
            num_data=base_instance.h1.size(0),
            num_features=base_instance.alpha_weights.size(0),
            hidden_nodes=base_instance.hidden_nodes,
            output_nodes=base_instance.beta_weights.size(1),
            activation_function=base_instance.activation_function.__class__.__name__,
            method=base_instance.method,
            # penalty_term=base_instance.penalty_term
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
            nn.Parameter(torch.zeros(self.n_hidden_nodes[1], self.extended_beta_weights.size(1)), requires_grad=True)
        )
        self.gamma_weights = (
            nn.Parameter(torch.zeros(self.extended_beta_weights.size(1), self.beta_weights.size(1)), requires_grad=True)
        )

    def create_hidden_layer(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Create a hidden layer with added noise based on the given weights.

        Args:
            weights (torch.Tensor): The weight tensor to create the hidden layer from.

        Returns:
            torch.Tensor: The created hidden layer with added noise.
        """

        return self._create_hidden_layer(weights, self.n_hidden_nodes[1])

    def _create_hidden_layer(self, weights: torch.Tensor, n_hidden_nodes: int) -> torch.Tensor:
        """
        Internal method to create a hidden layer with added noise.

        Args:
            weights (torch.Tensor): The weight tensor to create the hidden layer from.
            n_hidden_nodes (int): Number of hidden nodes in the layer.

        Returns:
            torch.Tensor: The created hidden layer with added noise.
        """

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
        """
        Train the next layer of the network.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.

        Returns:
            None
        """

        super(MultiPhaseDeepRandomizedNeuralNetworkSubsequent, self).train_ith_layer(
            train_loader=train_loader,
            hi=self.h2,
            weights1=self.extended_beta_weights,
            weights2=self.gamma_weights,
            method=self.method,
            hi_prev=self.h1
        )

    def predict_and_evaluate(
            self, dataloader, operation: str, layer_weights  = None,
            num_hidden_layers: int = None, verbose: bool = True):
        """
        Predict and evaluate the performance of the subsequent network layers.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
            operation (str): Operation being performed ('train' or 'test').
            layer_weights (torch.Tensor, optional): Weights for the layers. Defaults to None.
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to None.
            verbose (bool, optional): Whether to print detailed logs. Defaults to True.

        Returns:
            list: List containing accuracy, precision, recall, F1-score, and confusion matrix.
        """

        return super(MultiPhaseDeepRandomizedNeuralNetworkSubsequent, self).predict_and_evaluate(
            dataloader, operation, layer_weights, num_hidden_layers, verbose
        )

    def pruning(self, pruning_percentage: float, pruning_method: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prune the gamma weights of the network.

        Args:
            pruning_percentage (float): Percentage of weights to be pruned.
            pruning_method (str): Method to be used for pruning ('max_rank' or 'sum_weight').

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Indices of most important and least important weights.
        """

        return self.prune_weights(self.gamma_weights, pruning_percentage, pruning_method)


class MultiPhaseDeepRandomizedNeuralNetworkFinal(MultiPhaseDeepRandomizedNeuralNetworkSubsequent):
    def __init__(self, subsequent_instance: MultiPhaseDeepRandomizedNeuralNetworkSubsequent, mu: float, sigma: float):
        """
        Initialize the MultiPhaseDeepRandomizedNeuralNetworkFinal class.

        Args:
            subsequent_instance (MultiPhaseDeepRandomizedNeuralNetworkSubsequent): An instance of the subsequent class.
            mu (float): Mean for normal distribution to create noise.
            sigma (float): Standard deviation for normal distribution to create noise.
        """

        super(MultiPhaseDeepRandomizedNeuralNetworkFinal, self).__init__(
            subsequent_instance,
            mu,
            sigma
        )

        self.alpha_weights.data = subsequent_instance.alpha_weights.data.clone()
        self.beta_weights.data = subsequent_instance.beta_weights.data.clone()
        self.extended_beta_weights.data = subsequent_instance.extended_beta_weights.data.clone()
        self.gamma_weights.data = subsequent_instance.gamma_weights.data.clone()

        self.h1.data = subsequent_instance.h1.data.clone()
        self.h2.data = subsequent_instance.h2.data.clone()

        self.mu = mu
        self.sigma = sigma

        self.n_hidden_nodes = subsequent_instance.hidden_nodes
        self.method = subsequent_instance.method

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

        return self._create_hidden_layer(weights, self.n_hidden_nodes[2])

    def train_layer(self, train_loader):
        """
        Train the next layer of the network.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        """

        return super(MultiPhaseDeepRandomizedNeuralNetworkSubsequent, self).train_ith_layer(
            train_loader=train_loader,
            hi=self.h3,
            weights1=self.extended_gamma_weights,
            weights2=self.delta_weights,
            method=self.method,
            hi_prev=self.h2
        )

    def predict_and_evaluate(
            self, dataloader, operation: str, layer_weights: torch.Tensor = None, num_hidden_layers: int = None,
            verbose: bool = True) -> list:
        """
        Predict and evaluate the performance of the final network layers.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
            operation (str): Operation being performed ('train' or 'test').
            layer_weights (torch.Tensor, optional): Weights for the layers. Defaults to None.
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to None.
            verbose (bool, optional): Whether to print detailed logs. Defaults to True.

        Returns:
            list: List containing accuracy, precision, recall, F1-score, and confusion matrix.
        """

        return super(MultiPhaseDeepRandomizedNeuralNetworkSubsequent, self).predict_and_evaluate(
            dataloader, operation, layer_weights, num_hidden_layers, verbose
        )
