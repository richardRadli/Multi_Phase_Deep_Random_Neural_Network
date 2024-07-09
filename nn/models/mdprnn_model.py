import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score
from tqdm import tqdm


class MultiPhaseDeepRandomizedNeuralNetwork(nn.Module):
    def __init__(self, num_data, num_features, hidden_nodes, output_nodes, activation_function):
        super(MultiPhaseDeepRandomizedNeuralNetwork, self).__init__()
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

    def pruning(self, pruning_percentage: float):
        abs_beta_weights = torch.abs(self.beta_weights)
        importance_score = torch.sum(abs_beta_weights, dim=1)
        num_neurons_to_prune = int(pruning_percentage * abs_beta_weights.shape[0])
        least_important_prune_indies = torch.argsort(importance_score)[:num_neurons_to_prune]
        most_important_prune_indies = torch.argsort(importance_score, descending=True)[:num_neurons_to_prune]

        return most_important_prune_indies, least_important_prune_indies

        # ranks = torch.zeros_like(abs_beta_weights, dtype=torch.long)
        #
        # for col in range(abs_beta_weights.shape[1]):
        #     col_values = abs_beta_weights[:, col]
        #     sorted_indices = torch.argsort(col_values)
        #
        #     for rank, index in enumerate(sorted_indices):
        #         ranks[index, col] = rank + 1
        #
        # for col in range(ranks.shape[1]):
        #     # Get the indices where the rank is greater than 90
        #     prune_indices = ranks[:, col] > 90
        #
        #     # Zero out the corresponding columns in the alpha weights
        #     self.alpha_weights.data[:, prune_indices] = 0

    def predict_and_evaluate(self, dataloader, operation: str):
        for x, y in tqdm(dataloader, total=len(dataloader), desc=f"Predicting {operation} set"):
            h1 = self.activation_function(x @ self.alpha_weights)
            predictions = self.forward(h1)

            y_predicted_argmax = torch.argmax(predictions, dim=-1)
            y_true_argmax = torch.argmax(y, dim=-1)
            accuracy = accuracy_score(y_true_argmax, y_predicted_argmax)

            print(f"Accuracy of {operation} set: {accuracy:.4f}")

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
