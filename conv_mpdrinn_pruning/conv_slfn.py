import torch
import torch.nn as nn
import logging

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utils.utils import measure_execution_time


class ConvolutionalSLFN(nn.Module):
    def __init__(self, parameter_settings: dict, dataset_info: dict):
        super(ConvolutionalSLFN, self).__init__()

        # Parameter settings
        self.parameter_settings = parameter_settings
        self.num_filters = self.parameter_settings["num_filters"]
        self.kernel_size = self.parameter_settings["conv_kernel_size"]
        self.stride = self.parameter_settings["stride"]
        self.padding = self.parameter_settings["padding"]
        self.pool_kernel_size = self.parameter_settings["pool_kernel_size"]
        self.rcond = self.parameter_settings["rcond"]
        self.pruning_percentage = self.parameter_settings["pruning_percentage"]

        # Dataset info
        self.dataset_info = dataset_info
        self.output_channels = dataset_info["out_channels"]
        self.width = dataset_info["width"]
        self.height = dataset_info["height"]
        self.num_train_images = dataset_info["num_train_images"]

        self.num_hidden_neurons = self.calculate_flattened_size(self.width, self.height)
        self.accuracy = []

        logging.info(f"Number of neurons: {self.num_hidden_neurons}")

        self.conv1 = (
            nn.Conv2d(
                in_channels=dataset_info["in_channels"],
                out_channels=self.num_filters,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=False
            )
        )

        with torch.no_grad():
            nn.init.orthogonal_(self.conv1.weight)

        self.pooling = nn.MaxPool2d(kernel_size=self.pool_kernel_size)
        self.hidden = nn.Flatten()
        self.h1 = (
            nn.Parameter(
                torch.zeros(self.num_train_images, self.num_hidden_neurons), requires_grad=True
            )
        )
        self.beta_weights = (
            nn.Parameter(torch.zeros(self.num_hidden_neurons, self.output_channels), requires_grad=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pooling(x)
        x = self.hidden(x)
        return x

    @measure_execution_time
    def train_net(self, train_loader):
        for _, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            with torch.no_grad():
                self.h1.data = self.forward(images)

            targets = torch.eye(self.output_channels)[labels]
            self.compute_beta(target=targets)

    def test_net(self, base_instance, data_loader, layer_weights, num_layers):
        for _, (images, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
            labels = torch.eye(self.output_channels)[labels]
            with torch.no_grad():
                h1 = base_instance.forward(images)

            if num_layers == 1:
                predictions = self.predict(h1, layer_weights)
            elif num_layers == 2:
                h2 = torch.relu(h1.matmul(layer_weights[0]))
                predictions = self.predict(h2, layer_weights[1])
            elif num_layers == 3:
                h2 = torch.relu(h1.matmul(layer_weights[0]))
                h3 = torch.relu(h2.matmul(layer_weights[1]))
                predictions = self.predict(h3, layer_weights[2])
            else:
                raise ValueError(f"Unexpected number of layers: {num_layers}")

            y_predicted_argmax = torch.argmax(predictions, dim=-1)
            y_true_argmax = torch.argmax(labels, dim=-1)

            accuracy = accuracy_score(y_true_argmax, y_predicted_argmax)
            self.accuracy.append(accuracy)
            logging.info(f"Accuracy: {accuracy * 100:.2f}%")

    def calculate_flattened_size(self, height, width):
        conv_height = ((height - self.kernel_size + 2 * self.padding) // self.stride) + 1
        conv_width = ((width - self.kernel_size + 2 * self.padding) // self.stride) + 1

        pool_height = conv_height // self.pool_kernel_size
        pool_width = conv_width // self.pool_kernel_size

        return pool_height * pool_width * self.num_filters

    def compute_beta(self, target):
        hidden_pinv = torch.linalg.pinv(self.h1.data, rcond=self.rcond)
        self.beta_weights.data = hidden_pinv.matmul(target)

    @staticmethod
    def predict(hidden_features, weights):
        return hidden_features.matmul(weights)

    def prune_conv_weights(self, weights: torch.Tensor, num_neurons, pruning_percentage: float) \
            -> tuple[torch.Tensor, torch.Tensor]:
        grouped_weights = weights.view(self.num_filters, num_neurons, -1)
        logging.info(grouped_weights.shape)
        importance_scores = torch.sum(torch.abs(grouped_weights), dim=[1, 2])
        num_filters_to_prune = int(pruning_percentage * self.num_filters)
        least_important_prune_indices = torch.argsort(importance_scores)[:num_filters_to_prune]
        most_important_prune_indices = torch.argsort(importance_scores, descending=True)[:num_filters_to_prune]

        return most_important_prune_indices, least_important_prune_indices

    def pruning_first_layer(self):
        num_neurons = int(self.beta_weights.size(0) / self.num_filters)
        most_important_filters, least_important_filters = (
            self.prune_conv_weights(
                self.beta_weights, num_neurons, self. pruning_percentage
            )
        )

        return most_important_filters, least_important_filters

    @staticmethod
    def replace_filters(main_net, aux_net, least_important_indices, most_important_indices):
        aux_weights = aux_net.conv1.weight.data[most_important_indices]
        main_net.conv1.weight.data[least_important_indices] = aux_weights
        return main_net


class SecondLayer(ConvolutionalSLFN):
    def __init__(self, base_instance):
        super(SecondLayer, self).__init__(
            parameter_settings=base_instance.parameter_settings,
            dataset_info=base_instance.dataset_info
        )

        self.mu = base_instance.parameter_settings["mu"]
        self.sigma = base_instance.parameter_settings["sigma"]
        self.n_hidden_nodes = base_instance.parameter_settings["hidden_nodes"][0]
        self.penalty_term = base_instance.parameter_settings["penalty_term"]

        self.h1.data = base_instance.h1.data.clone()
        self.beta_weights.data = base_instance.beta_weights.data.clone()

        self.extended_beta_weights = self.create_hidden_layer(self.beta_weights)
        self.h2 = (
            nn.Parameter(torch.zeros(self.n_hidden_nodes, self.extended_beta_weights.size(1)), requires_grad=True)
        )
        self.gamma_weights = (
            nn.Parameter(torch.zeros(self.extended_beta_weights.size(1), self.beta_weights.size(1)), requires_grad=True)
        )

    def create_hidden_layer(self, weights: torch.Tensor) -> torch.Tensor:
        return self._create_hidden_layer(weights, self.n_hidden_nodes)

    def _create_hidden_layer(self, weights: torch.Tensor, n_hidden_nodes: int) -> torch.Tensor:
        noise = torch.normal(mean=self.mu, std=self.sigma, size=(weights.shape[0], weights.shape[1]))
        w_rnd_out_i = weights + noise
        hidden_layer_i_a = torch.hstack((weights, w_rnd_out_i))

        w_rnd = torch.normal(mean=self.mu, std=self.sigma, size=(weights.shape[0], n_hidden_nodes // 2))
        q, _ = torch.linalg.qr(w_rnd)
        hidden_layer_i = torch.cat((hidden_layer_i_a, q), dim=1)
        return hidden_layer_i

    @measure_execution_time
    def train_fc_layer(self, train_loader):
        return (
            self.train_ith_layer(
                train_loader,
                self.h1,
                self.h2,
                self.extended_beta_weights,
                self.gamma_weights
            )
        )

    def train_ith_layer(self, train_loader, hi_prev, hi, weights1, weights2):
        for _, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            labels = torch.eye(self.output_channels)[labels]

            hi.data = torch.relu(hi_prev @ weights1)
            identity_matrix = torch.eye(hi.shape[1], device=hi.device)

            if hi.shape[0] > hi.shape[1]:
                weights2.data = (
                        torch.linalg.pinv(hi.T @ hi + identity_matrix / self.penalty_term, rcond=self.rcond)
                        @ (hi.T @ labels)
                )
            else:
                weights2.data = (
                        hi.T @ torch.linalg.pinv(hi @ hi.T + identity_matrix / self.penalty_term,
                                                 rcond=self.rcond) @ labels
                )

    def test_net(self, base_instance, data_loader, layer_weights, num_layers):
        return super(SecondLayer, self).test_net(
            base_instance, data_loader, layer_weights, num_layers
        )

    @staticmethod
    def prune_weights(weights: torch.Tensor, pruning_percentage: float) -> tuple[torch.Tensor, torch.Tensor]:
        abs_weights = torch.abs(weights)

        ranking_matrix = abs_weights.argsort(dim=0).argsort(dim=0)
        importance_score, not_used = torch.max(ranking_matrix, dim=1)

        num_neurons_to_prune = int(pruning_percentage * abs_weights.shape[0])
        least_important_prune_indices = torch.argsort(importance_score)[:num_neurons_to_prune]
        most_important_prune_indices = torch.argsort(importance_score, descending=True)[:num_neurons_to_prune]

        return most_important_prune_indices, least_important_prune_indices

    def pruning(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.prune_weights(self.gamma_weights, self.pruning_percentage)


class ThirdLayer(SecondLayer):
    def __init__(self, second_layer):
        super(ThirdLayer, self).__init__(
            second_layer
        )

        self.h2.data = second_layer.h2.data.clone()
        self.extended_beta_weights.data = second_layer.extended_beta_weights.data.clone()
        self.gamma_weights.data = second_layer.gamma_weights.data.clone()

        self.n_hidden_nodes = second_layer.parameter_settings["hidden_nodes"][1]

        self.extended_gamma_weights = self.create_hidden_layer(self.gamma_weights)
        self.h3 = nn.Parameter(torch.randn(self.h2.size()), requires_grad=True)
        self.delta_weights = nn.Parameter(torch.randn(self.h2.size(1), self.output_channels), requires_grad=True)

    @measure_execution_time
    def train_fc_layer(self, train_loader):
        """
        Train the next layer of the network.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        """

        return super(ThirdLayer, self).train_ith_layer(
            train_loader=train_loader,
            hi_prev=self.h2,
            hi=self.h3,
            weights1=self.extended_gamma_weights,
            weights2=self.delta_weights,
        )

    def test_net(self, base_instance, data_loader, layer_weights, num_layers):
        return super(ThirdLayer, self).test_net(
            base_instance, data_loader, layer_weights, num_layers
        )

    def pruning(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.prune_weights(self.delta_weights, self.pruning_percentage)
