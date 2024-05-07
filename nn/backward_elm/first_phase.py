import torch
import torch.nn.init as init

from tqdm import tqdm
from sklearn.metrics import accuracy_score

from config.config import BWELMConfig


class FirstPhase:
    def __init__(self,
                 n_input_nodes: int,
                 n_hidden_nodes: int,
                 train_loader,
                 test_loader,
                 ):
        self.cfg = BWELMConfig().parse()

        if self.cfg.seed:
            torch.manual_seed(1234)

        self.n_input_nodes = n_input_nodes
        self.n_hidden_nodes = n_hidden_nodes

        self.train_loader = train_loader
        self.test_loader = test_loader

    @staticmethod
    def inv_leaky_ReLU(x, alpha=0.2):
        return torch.where(x < 0, x * (1 / alpha), x)

    @staticmethod
    def get_weight_initialization(weight_init_type: str, empty_weight_matrix):
        weight_dict = {
            'uniform_0_1': init.uniform_(empty_weight_matrix, 0, 1),
            'uniform_1_1': init.uniform_(empty_weight_matrix, -1, 1),
            "xavier": init.xavier_uniform_(empty_weight_matrix),
            "relu": init.kaiming_uniform_(empty_weight_matrix, nonlinearity="relu"),
            "orthogonal": init.orthogonal_(empty_weight_matrix)
        }

        return weight_dict[weight_init_type]

    def calculate_beta_weights(self):
        empty_tensor = torch.empty((self.n_input_nodes, self.n_hidden_nodes))
        alpha_weights = self.get_weight_initialization(self.cfg.init_type, empty_tensor)

        for train_x, train_y in tqdm(self.train_loader):
            h1 = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)(train_x.matmul(alpha_weights))
            beta_weights = torch.pinverse(h1).matmul(train_y)

            return h1, alpha_weights, beta_weights

    def train_accuracy(self, h1, beta_weights):
        for _, train_y in tqdm(self.train_loader):
            predictions = h1.matmul(beta_weights)
            y_predicted_argmax = torch.argmax(predictions, dim=-1)
            y_true_argmax = torch.argmax(train_y, dim=-1)
            accuracy = accuracy_score(y_true_argmax, y_predicted_argmax)
            print(accuracy)

    def calculate_updated_alpha(self, beta_weights, sigma_beta: float = 0.09, sigma_h_est: float = 0.09):
        for train_x, train_y in tqdm(self.train_loader):
            beta_weights += torch.normal(0, sigma_beta, beta_weights.shape)
            h_est = self.inv_leaky_ReLU(train_y.matmul(torch.pinverse(beta_weights)))
            h_est += torch.normal(0, sigma_h_est, h_est.shape)
            updated_alpha_weights = torch.pinverse(train_x).matmul(h_est)

            return updated_alpha_weights

    def accuracy_of_ELM_with_updated_alpha(self, beta_weights, updated_alpha_weights):
        for train_x, train_y in tqdm(self.train_loader):
            h1_updated = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)(train_x.matmul(updated_alpha_weights))
            y_predicted_updated = h1_updated.matmul(beta_weights)
            y_predicted_updated_argmax = torch.argmax(y_predicted_updated, dim=-1)
            y_true_argmax = torch.argmax(train_y, dim=-1)
            accuracy = accuracy_score(y_true_argmax, y_predicted_updated_argmax)
            print(accuracy)

    def calculate_updated_beta(self, updated_alpha_weights, ort_alpha_weights: bool):
        if ort_alpha_weights:
            temp1 = updated_alpha_weights[:, updated_alpha_weights.shape[1] // 2:]
            temp2 = updated_alpha_weights[:, :updated_alpha_weights.shape[1] // 2]
            temp3 = torch.empty(temp1.shape)
            orthogonal_alpha = init.orthogonal_(temp3)
            updated_alpha_weights = torch.cat((orthogonal_alpha, temp2), dim=1)

        for train_x, train_y in tqdm(self.train_loader):
            h1_updated = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)(train_x.matmul(updated_alpha_weights))
            updated_beta_weights = torch.pinverse(h1_updated).matmul(train_y)
            return h1_updated, updated_beta_weights

    def main(self):
        h1, alpha_weights, beta_weights = self.calculate_beta_weights()
        self.train_accuracy(h1, beta_weights)
        updated_alpha_weights = self.calculate_updated_alpha(beta_weights, sigma_beta=0.5, sigma_h_est=0.5)
        self.accuracy_of_ELM_with_updated_alpha(beta_weights, updated_alpha_weights)
        h1_updated, updated_beta_weights = (
            self.calculate_updated_beta(updated_alpha_weights, ort_alpha_weights=True)
        )
        self.train_accuracy(h1_updated, updated_beta_weights)
