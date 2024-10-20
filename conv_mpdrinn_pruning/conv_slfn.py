import torch
import torch.nn as nn
import logging

from sklearn.metrics import accuracy_score
from tqdm import tqdm


class ConvolutionalSLFN(nn.Module):
    def __init__(self, num_filters, kernel_size, stride, padding, pool_kernel_size, rcond, dataset_info: dict):
        super(ConvolutionalSLFN, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_kernel_size = pool_kernel_size
        self.rcond = rcond
        self.dataset_info = dataset_info

        self.output_channels = dataset_info["out_channels"]
        self.width = dataset_info["width"]
        self.height = dataset_info["height"]
        self.num_train_images = dataset_info["num_train_images"]
        self.num_hidden_neurons = self.calculate_flattened_size(self.width, self.height)

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
            else:
                raise ValueError(f"Unexpected number of layers: {num_layers}")

            y_predicted_argmax = torch.argmax(predictions, dim=-1)
            y_true_argmax = torch.argmax(labels, dim=-1)

            accuracy = accuracy_score(y_true_argmax, y_predicted_argmax)
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

    def prune_weights(self, weights: torch.Tensor, num_neurons, pruning_percentage: float) \
            -> tuple[torch.Tensor, torch.Tensor]:
        grouped_weights = weights.view(self.num_filters, num_neurons, -1)
        logging.info(grouped_weights.shape)
        importance_scores = torch.sum(torch.abs(grouped_weights), dim=[1, 2])
        num_filters_to_prune = int(pruning_percentage * self.num_filters)
        least_important_prune_indices = torch.argsort(importance_scores)[:num_filters_to_prune]
        most_important_prune_indices = torch.argsort(importance_scores, descending=True)[:num_filters_to_prune]

        return most_important_prune_indices, least_important_prune_indices

    def pruning(self, pruning_percentage):
        num_neurons = int(self.beta_weights.size(0) / self.num_filters)
        most_important_filters, least_important_filters = (
            self.prune_weights(
                self.beta_weights, num_neurons, pruning_percentage
            )
        )

        return most_important_filters, least_important_filters

    @staticmethod
    def replace_filters(main_net, aux_net, least_important_indices, most_important_indices):
        aux_weights = aux_net.conv1.weight.data[most_important_indices]
        main_net.conv1.weight.data[least_important_indices] = aux_weights
        return main_net


class MultiPhaseDeepRandomizedNeuralNetworkSubsequent(ConvolutionalSLFN):
    def __init__(self, base_instance, mu, sigma, n_hidden_nodes, penalty_term):
        super(MultiPhaseDeepRandomizedNeuralNetworkSubsequent, self).__init__(
            num_filters=base_instance.num_filters,
            kernel_size=base_instance.kernel_size,
            stride=base_instance.stride,
            padding=base_instance.padding,
            pool_kernel_size=base_instance.pool_kernel_size,
            rcond=base_instance.rcond,
            dataset_info=base_instance.dataset_info
        )

        self.mu = mu
        self.sigma = sigma
        self.n_hidden_nodes = n_hidden_nodes
        self.penalty_term = penalty_term

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

    def train_second_layer(self, train_loader):
        for _, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            labels = torch.eye(self.output_channels)[labels]

            self.h2.data = torch.relu(self.h1 @ self.extended_beta_weights)
            identity_matrix = torch.eye(self.h2.shape[1], device=self.h2.device)

            if self.h2.shape[0] > self.h2.shape[1]:
                self.gamma_weights.data = (
                        torch.linalg.pinv(self.h2.T @ self.h2 + identity_matrix / self.penalty_term, rcond=self.rcond)
                        @ (self.h2.T @ labels)
                )

            else:
                self.gamma_weights.data = (
                        self.h2.T @ torch.linalg.pinv(self.h2 @ self.h2.T + identity_matrix / self.penalty_term,
                                                      rcond=self.rcond) @ labels
                )

    def test_net(self, base_instance, data_loader, layer_weights, num_layers):
        return super(MultiPhaseDeepRandomizedNeuralNetworkSubsequent, self).test_net(
            base_instance, data_loader, layer_weights, num_layers
        )
