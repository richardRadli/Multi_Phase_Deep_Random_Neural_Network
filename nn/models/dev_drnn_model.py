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
        train_loader=None
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

        colorama.init()

        self.activation_function = self.get_activation(activation_function)
        self.train_loader = train_loader

        self.alpha_weights = nn.Parameter(torch.randn(num_features, hidden_nodes[0]), requires_grad=False)
        self.beta_weights = nn.Parameter(torch.zeros(hidden_nodes[0], output_nodes), requires_grad=False)
        self.h1 = nn.Parameter(torch.zeros(num_data, hidden_nodes[0]), requires_grad=False)
        self.bias = nn.Parameter(torch.randn(hidden_nodes[0]), requires_grad=False)

        self.predictions = None

        self.hidden_nodes = hidden_nodes
        self.rcond = rcond
        self.penalty_term = penalty_term

        self.condition_number_list = []

    def set_train_predictions(self, prediction):
        self.predictions = prediction

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
                self.condition_number_list.append(torch.linalg.cond(hi.data))
            else:
                hi.data = self.activation_function(hi_prev @ weights1)
                self.condition_number_list.append(torch.linalg.cond(hi.data))

            if hi.shape[0] > hi.shape[1]:
                identity_l = torch.eye(hi.shape[1])
                pseudo_inv_input = hi.T @ hi + identity_l / self.penalty_term

                if self.rcond is not None:
                    weights2.data = torch.linalg.pinv(pseudo_inv_input, rcond=self.rcond) @ (hi.T @ train_y)
                    self.condition_number_list.append(torch.linalg.cond(weights2.data))
                else:
                    weights2.data = torch.linalg.pinv(pseudo_inv_input) @ (hi.T @ train_y)
                    self.condition_number_list.append(torch.linalg.cond(weights2.data))
            else:
                identity_n = torch.eye(hi.shape[0])
                pseudo_inv_input = hi @ hi.T + identity_n / self.penalty_term

                if self.rcond is not None:
                    weights2.data = hi.T @ torch.linalg.pinv(pseudo_inv_input, rcond=self.rcond) @ train_y
                    self.condition_number_list.append(torch.linalg.cond(weights2.data))
                else:
                    weights2.data = hi.T @ torch.linalg.pinv(pseudo_inv_input) @ train_y
                    self.condition_number_list.append(torch.linalg.cond(weights2.data))

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
            self.set_train_predictions(predictions)
            # self.predictions = predictions

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
    def __init__(self,
                 first_layer_instance: DevDeepRandomizedNeuralNetworkFirstLayer,
                 mu: float,
                 sigma: float,
                 run_init: bool = True
    ):
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

        self.h2 = None
        self.extended_beta_weights = None
        self.class_direction_tensor = None
        self.allocation = None
        self.gamma_weights = None
        self.error_matrix = None
        self.alpha_weights.data = first_layer_instance.alpha_weights.data.clone()
        self.beta_weights.data = first_layer_instance.beta_weights.data.clone()
        self.h1.data = first_layer_instance.h1.data.clone()
        self.n_hidden_nodes = first_layer_instance.hidden_nodes
        self.predictions = first_layer_instance.predictions
        self.rcond = first_layer_instance.rcond

        self.condition_number_list = first_layer_instance.condition_number_list

        self.mu = mu
        self.sigma = sigma

        if run_init:
            self.setup_layer_logic()

    def setup_layer_logic(self):
        self.allocation, self.class_direction_tensor, self.error_matrix = (
            self.allocate_neurons_per_class_pair(
                hidden_layer=self.h1.data,
                total_neurons=self.n_hidden_nodes[1]
            )
        )

        self.extended_beta_weights = self.create_hidden_layer(
            self.beta_weights
        )

        self.h2 = nn.Parameter(
            torch.zeros(
                self.n_hidden_nodes[1],
                self.extended_beta_weights.size(1)
            ),
            requires_grad=False
        )
        self.gamma_weights = nn.Parameter(
            torch.zeros(
                self.extended_beta_weights.size(1),
                self.beta_weights.size(1)
            ),
            requires_grad=False
        )

    def _get_class_stats(self, hidden_layer):
        _, y_one_hot = next(iter(self.train_loader))
        y_true = torch.argmax(y_one_hot, dim=-1)
        y_pred = torch.argmax(self.predictions, dim=-1)
        num_classes = y_one_hot.size(1)

        # class means in the hidden layer
        classes = torch.unique(y_true).sort()[0]
        means = torch.stack([hidden_layer[y_true == c].mean(dim=0) for c in classes])

        # direction vectors and Euc. dist.
        direction_tensor = means.unsqueeze(1) - means.unsqueeze(0)
        dist_matrix = torch.norm(direction_tensor, dim=-1, p=2)

        # confusion matrix
        conf = torch.zeros((num_classes, num_classes))
        for t, p in zip(y_true, y_pred):
            conf[t, p] += 1

        # Select only the badly predicted cases
        error_mask = (conf + conf.T > 0).float()
        direction_tensor = direction_tensor * error_mask.unsqueeze(-1)

        # Normalized error matrix
        error_matrix = conf / (conf.sum(dim=1, keepdim=True) + 1e-8)

        return direction_tensor, dist_matrix, error_matrix

    def _allocate_neurons_per_class_pair(self, hidden_layer, total_neurons: int, eps: float = 1e-8):
        dir_tensor, dist_mtx, err_mtx = self._get_class_stats(hidden_layer)

        combined_err = (err_mtx + err_mtx.T)
        difficulty_score = combined_err / (dist_mtx + eps)

        C = difficulty_score.shape[0]
        triu_idx = torch.triu_indices(C, C, offset=1)

        all_scores = difficulty_score[triu_idx[0], triu_idx[1]]
        error_mask = all_scores > 0

        active_indices = torch.where(error_mask)[0]
        scores = all_scores[active_indices]

        if len(active_indices) == 0:
            return {}, dir_tensor

        sorted_sub_indices = torch.argsort(scores, descending=True)
        num_active_pairs = len(active_indices)
        allocation = {}

        for idx in active_indices:
            c1, c2 = triu_idx[0][idx].item(), triu_idx[1][idx].item()
            allocation[(int(c1), int(c2))] = 1

        remaining = total_neurons - num_active_pairs
        if remaining > 0:
            weights = scores / (scores.sum() + eps)
            extra_neurons = torch.floor(weights * remaining).int()

            for i, sub_idx in enumerate(sorted_sub_indices):
                actual_idx = active_indices[sub_idx]
                c1, c2 = triu_idx[0][actual_idx].item(), triu_idx[1][actual_idx].item()
                allocation[(int(c1), int(c2))] += extra_neurons[sub_idx].item()

        diff = total_neurons - sum(allocation.values())
        for i in range(abs(diff)):
            sub_idx = sorted_sub_indices[i % num_active_pairs]
            actual_idx = active_indices[sub_idx]
            pair = int(triu_idx[0][actual_idx]), int(triu_idx[1][actual_idx])
            allocation[pair] += 1 if diff > 0 else -1

        return allocation, dir_tensor, err_mtx

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

    def _create_hidden_layer_new(
            self,
            weights: torch.Tensor,
            eps: float = 1e-8,
            scaling_factor: bool = False,
            alpha: float = 0.1,
            orthogonal=False
    ):
        dimension, _ = weights.shape

        noise = torch.normal(mean=self.mu, std=self.sigma, size=weights.shape)
        w_rnd_out_i = weights + noise

        new_columns = []

        for (c1, c2), n_neurons in self.allocation.items():
            # base direction from class means
            v_base = self.class_direction_tensor[c1, c2]
            v_unit = v_base / (v_base.norm() + eps)

            for i in range(n_neurons):
                if i == 0:
                    v = v_unit
                else:
                    v_noise = torch.normal(mean=0.0, std=self.sigma, size=(dimension,))
                    v = v_unit + v_noise
                    v = v / (v.norm() + eps)

                new_columns.append(v.view(dimension, 1))

        if not new_columns:
            raise ValueError("new_columns is empty!")

        new_columns_matrix = torch.cat(new_columns, dim=1)

        if scaling_factor:
            new_columns_matrix = new_columns_matrix * alpha

        if orthogonal:
            ort = new_columns_matrix.clone()
            torch.nn.init.orthogonal_(ort)
            ort = ort @ ort.t()
            hidden_layer = torch.cat([weights, w_rnd_out_i, ort], dim=1)
        else:
           hidden_layer = torch.cat([weights, w_rnd_out_i, new_columns_matrix], dim=1)

        return hidden_layer

    def _create_hidden_layer_baseline(self, weights, n_hidden_nodes):
        noise = torch.normal(mean=self.mu, std=self.sigma, size=(weights.shape[0], weights.shape[1]))
        w_rnd_out_i = weights + noise
        hidden_layer_i_a = torch.hstack((weights, w_rnd_out_i))

        w_rnd = torch.normal(mean=self.mu, std=self.sigma, size=(weights.shape[0], n_hidden_nodes))
        q = w_rnd.clone()
        torch.nn.init.orthogonal_(q)
        q = q @ q.t()

        # plot_vector_diversity([q], "Old method")
        # plot_neuron_vectors_3d([q])

        hidden_layer_i = torch.cat((hidden_layer_i_a, q), dim=1)

        return hidden_layer_i

    def _create_hidden_layer_threshold(self, weights, error_threshold: float, similarity_threshold: float):
        dimension, _ = weights.shape
        new_columns = []

        total_target_neurons = sum(self.allocation.values())
        allocated_directed_neurons = 0

        _, y_one_hot = next(iter(self.train_loader))
        y_true = torch.argmax(y_one_hot, dim=-1)
        y_pred = torch.argmax(self.predictions, dim=-1)
        misclassified_mask = (y_pred != y_true)

        for (c1, c2), n_neurons in self.allocation.items():
            error_rate = self.error_matrix[c1, c2]

            if error_rate > error_threshold:
                mask_c1_c2 = (y_true == c1) & (y_pred == c2) & misclassified_mask

                if mask_c1_c2.any():
                    v_base = self.class_direction_tensor[c1, c2]
                    v_unit = v_base / (v_base.norm() + 1e-8)

                    for i in range(n_neurons):
                        if i == 0:
                            v = v_unit
                        else:
                            v_noise = torch.normal(mean=0.0, std=self.sigma, size=(dimension,))
                            v = (v_unit + v_noise).div((v_unit + v_noise).norm() + 1e-8)

                        v_reshaped = v.view(dimension, 1)
                        is_redundant = False

                        if len(new_columns) > 0:
                            current_pool = torch.cat(new_columns, dim=1)
                            similarities = torch.mm(current_pool.T, v_reshaped).view(-1)

                            if torch.any(similarities > similarity_threshold):
                                is_redundant = True

                        if not is_redundant:
                            new_columns.append(v_reshaped)
                            allocated_directed_neurons += 1

        remaining_neurons = total_target_neurons - len(new_columns)
        if remaining_neurons > 0:
            random_mtx = torch.randn(dimension, remaining_neurons)
            q_random = torch.nn.init.orthogonal_(random_mtx)

            for i in range(remaining_neurons):
                new_columns.append(q_random[:, i].view(dimension, 1))

        new_columns_matrix = torch.cat(new_columns, dim=1)
        # new_columns_matrix = new_columns_matrix @ new_columns_matrix.t()
        hidden_layer = torch.cat([weights, new_columns_matrix], dim=1)

        return hidden_layer

    def _create_hidden_layer_residual(self, weights, hidden_layer, n_hidden_nodes, rcond):
        noise = torch.normal(mean=self.mu, std=self.sigma, size=(weights.shape[0], weights.shape[1]))
        w_rnd_out_i = weights + noise

        _, y_one_hot = next(iter(self.train_loader))
        y_true = torch.argmax(y_one_hot, dim=1)
        y_pred = torch.argmax(self.predictions, dim=1)

        misclassified_mask = (y_pred != y_true)
        hidden_misclassified = hidden_layer[misclassified_mask]
        residual_misclassified = (y_one_hot - self.predictions)[misclassified_mask]

        if hidden_misclassified.size(0) > 0:
            beta_res = torch.linalg.pinv(hidden_misclassified, rcond=rcond) @ residual_misclassified
        else:
            beta_res = torch.zeros((hidden_layer.size(1), y_one_hot.size(1)), device=hidden_layer.device)

        remaining_neurons = n_hidden_nodes - (weights.shape[1] + w_rnd_out_i.shape[1] + beta_res.shape[1])

        w_rnd = torch.normal(mean=self.mu, std=self.sigma, size=(weights.shape[0], remaining_neurons))
        q = w_rnd.clone()
        torch.nn.init.orthogonal_(q)
        # q = q @ q.t()
        hidden_layer = torch.cat([weights, w_rnd_out_i, beta_res, q], dim=1)

        return hidden_layer

    def create_hidden_layer(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Create a hidden layer with added noise based on the given weights.

        Args:
            weights (torch.Tensor): The weight tensor to create the hidden layer from.

        Returns:
            torch.Tensor: The created hidden layer with added noise.
        """

        # return self._create_hidden_layer_baseline(weights, n_hidden_nodes=self.hidden_nodes[1])
        # return self._create_hidden_layer_new(weights, orthogonal=False, alpha=True)
        # return self._create_hidden_layer_threshold(weights, 1e-8)
        return self._create_hidden_layer_residual(weights, self.h1.data, self.hidden_nodes[1], self.rcond)

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


class DevDeepRandomizedNeuralNetworkThirdLayer(DevDeepRandomizedNeuralNetworkSecondLayer):
    def __init__(
            self,
            second_layer_instance: DevDeepRandomizedNeuralNetworkSecondLayer
    ):
        super(DevDeepRandomizedNeuralNetworkThirdLayer, self).__init__(
            second_layer_instance,
            second_layer_instance.mu,
            second_layer_instance.sigma,
            run_init=False
        )

        self.extended_beta_weights = nn.Parameter(
            second_layer_instance.extended_beta_weights.data.clone(),
            requires_grad=False
        )
        self.gamma_weights = nn.Parameter(
            second_layer_instance.gamma_weights.data.clone(),
            requires_grad=False
        )

        self.h1 = nn.Parameter(
            second_layer_instance.h1.data.clone(),
            requires_grad=False
        )
        self.h2 = nn.Parameter(
            second_layer_instance.h2.data.clone(),
            requires_grad=False
        )

        self.allocation, self.class_direction_tensor, self.error_matrix = self.allocate_neurons_per_class_pair(
            hidden_layer=self.h2.data,
            total_neurons=self.n_hidden_nodes[2]
        )

        self.extended_gamma_weights = self.create_hidden_layer(
            self.gamma_weights
        )

        self.h3 = nn.Parameter(
            torch.zeros(
                self.n_hidden_nodes[2],
                self.extended_gamma_weights.size(1)
            ),
            requires_grad=False
        )

        self.delta_weights = nn.Parameter(
            torch.zeros(
                self.extended_gamma_weights.size(1),
                self.beta_weights.size(1)
            ),
            requires_grad=False
        )

    def create_hidden_layer(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Create a hidden layer with added noise based on the given weights.

        Args:
            weights (torch.Tensor): The weight tensor to create the hidden layer from.

        Returns:
            torch.Tensor: The created hidden layer with added noise.
        """

        # return self._create_hidden_layer_baseline(weights, n_hidden_nodes=self.hidden_nodes[2])
        # return self._create_hidden_layer_new(weights, orthogonal=False, alpha=True)
        # return self._create_hidden_layer_threshold(weights, 1e-8)
        return self._create_hidden_layer_residual(weights, self.h2.data, self.hidden_nodes[2], rcond=self.rcond)

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