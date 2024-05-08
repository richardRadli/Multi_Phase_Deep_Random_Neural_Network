import logging
import pandas as pd
import torch
import torch.nn.init as init

from sklearn.metrics import accuracy_score

from config.config import BWELMConfig
from nn.models.elm import ELM


class FirstPhase:
    def __init__(self,
                 n_input_nodes: int,
                 n_hidden_nodes: int,
                 sigma: float,
                 train_loader,
                 test_loader,
                 ):
        self.cfg = BWELMConfig().parse()

        self.n_input_nodes = n_input_nodes
        self.n_hidden_nodes = n_hidden_nodes
        self.sigma = sigma

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.elm = ELM(self.cfg.activation_function)
        self.reverse_elm = ELM(self.cfg.inverse_activation_function)

        self.training_accuracies = []
        self.test_accuracies = []

    @staticmethod
    def get_weight_initialization(weight_init_type: str, empty_weight_matrix):
        """

        Args:
            weight_init_type:
            empty_weight_matrix:

        Returns:

        """

        weight_dict = {
            'uniform_0_1': init.uniform_(empty_weight_matrix, 0, 1),
            'uniform_1_1': init.uniform_(empty_weight_matrix, -1, 1),
            "xavier": init.xavier_uniform_(empty_weight_matrix),
            "relu": init.kaiming_uniform_(empty_weight_matrix),
            "orthogonal": init.orthogonal_(empty_weight_matrix)
        }

        return weight_dict[weight_init_type]

    @staticmethod
    def calculate_condition(weights):
        _, s, _ = torch.svd(weights)
        condition = torch.max(s) / torch.min(s)
        return condition

    def forward_weight_calculations(self, weights):
        """

        Args:
            weights:

        Returns:

        """

        for train_x, train_y in self.train_loader:
            hidden_layer = self.elm(train_x, weights)
            beta_weights = torch.pinverse(hidden_layer).matmul(train_y)

            return beta_weights

    def calculate_accuracy(self, operation, alpha_weights, beta_weights):
        """

        Args:
            operation:
            alpha_weights:
            beta_weights:

        Returns:

        """

        if operation not in ["train", "test"]:
            raise ValueError('An unknown operation \'%s\'.' % operation)

        if operation == "train":
            dataloader = self.train_loader
        else:
            dataloader = self.test_loader

        for x, y in dataloader:
            h1 = self.elm(x, alpha_weights)
            predictions = h1.matmul(beta_weights)
            y_predicted_argmax = torch.argmax(predictions, dim=-1)
            y_true_argmax = torch.argmax(y, dim=-1)
            accuracy = accuracy_score(y_true_argmax, y_predicted_argmax)

            self.training_accuracies.append(accuracy) if operation == "train" \
                else self.test_accuracies.append(accuracy)

    def calculate_updated_alpha(self, beta_weights, sigma_beta, sigma_h_est):
        """

        Args:
            beta_weights:
            sigma_beta:
            sigma_h_est:

        Returns:

        """

        for train_x, train_y in self.train_loader:
            beta_weights += torch.normal(0, sigma_beta, beta_weights.shape)
            h_est = self.reverse_elm(train_y, torch.pinverse(beta_weights))
            h_est += torch.normal(0, sigma_h_est, h_est.shape)
            updated_alpha_weights = torch.pinverse(train_x).matmul(h_est)

            return updated_alpha_weights

    @staticmethod
    def orthogonalize_weights(weights):
        num_rows, num_cols = weights.size()
        first_half_cols = num_cols // 2
        first_half = weights[:, :first_half_cols]
        second_half = weights[:, first_half_cols:]

        u, _, v = torch.svd(first_half)
        orthogonalized_weights = torch.matmul(u, v.t())

        new_matrix = torch.cat((orthogonalized_weights, second_half), dim=1)

        return new_matrix

    def plot_accuracies(self):
        data = {
            'training_accuracy': self.training_accuracies,
            'test_accuracy': self.test_accuracies
        }

        index = ["forward1", "backward", "forward2"]

        df = pd.DataFrame(data, index=index)
        logging.info(df)

    def first_forward(self):
        """

        Returns:

        """

        empty_tensor = torch.empty((self.n_input_nodes, self.n_hidden_nodes), dtype=torch.float)
        alpha_weights = self.get_weight_initialization(self.cfg.init_type, empty_tensor)
        beta_weights = self.forward_weight_calculations(alpha_weights)
        self.calculate_accuracy("train", alpha_weights, beta_weights)
        self.calculate_accuracy("test", alpha_weights, beta_weights)

        self.calculate_condition(alpha_weights)
        self.calculate_condition(beta_weights)

        return beta_weights

    def backward(self, beta_weights, orthogonal_weights: bool):
        """

        Args:
            beta_weights:
            orthogonal_weights:

        Returns:

        """

        updated_alpha_weights = self.calculate_updated_alpha(beta_weights,
                                                             sigma_beta=self.sigma,
                                                             sigma_h_est=self.sigma)
        if orthogonal_weights:
            updated_alpha_weights = self.orthogonalize_weights(updated_alpha_weights)

        self.calculate_accuracy("train", updated_alpha_weights, beta_weights)
        self.calculate_accuracy("test", updated_alpha_weights, beta_weights)
        self.calculate_condition(updated_alpha_weights)

        return updated_alpha_weights

    def second_forward(self, updated_alpha_weights):
        """

        Args:
            updated_alpha_weights:

        Returns:

        """

        updated_beta_weights = self.forward_weight_calculations(updated_alpha_weights)
        self.calculate_accuracy("train", updated_alpha_weights, updated_beta_weights)
        self.calculate_accuracy("test", updated_alpha_weights, updated_beta_weights)

    def main(self):
        # Forward 1
        beta_weights = self.first_forward()

        # Backward
        updated_alpha_weights = self.backward(beta_weights, orthogonal_weights=False)

        # Forward 2
        self.second_forward(updated_alpha_weights)

        self.plot_accuracies()
