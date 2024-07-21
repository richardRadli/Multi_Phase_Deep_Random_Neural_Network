import colorama
import logging
import os
import numpy as np
import scipy.linalg as linalg

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nn.dataloaders.npz_dataloader import NpzDataset
from torch.utils.data import DataLoader
from typing import List, Tuple
from tqdm import tqdm

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import general_dataset_configs, helm_paths_config
from utils.utils import (create_timestamp, load_config_json, setup_logger, measure_execution_time,
                         reorder_metrics_lists, insert_data_to_excel, average_columns_in_excel)


class HELM:
    def __init__(self):
        timestamp = create_timestamp()
        setup_logger()

        self.cfg = (
            load_config_json(json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_helm"),
                             json_filename=JSON_FILES_PATHS.get_data_path("config_helm"))
        )

        dataset_name = self.cfg.get("dataset_name")
        helm_config = helm_paths_config(dataset_name)

        self.filename = \
            os.path.join(helm_config.get("path_to_results"), f"{timestamp}_{dataset_name}_dataset.xlsx")

        if self.cfg.get("seed"):
            np.random.seed(self.cfg.get("seed"))

        file_path = general_dataset_configs(dataset_name).get("cached_dataset_file")
        self.train_loader, _, self.test_loader = (
            self.create_train_valid_test_datasets(file_path)
        )

        self.num_features = general_dataset_configs(dataset_name).get("num_features")

        colorama.init()

        self.C_penalty = self.cfg.get("penalty")
        self.scaling_factor = self.cfg.get("scaling_factor")

    @staticmethod
    def create_train_valid_test_datasets(file_path):
        train_dataset = NpzDataset(file_path, operation="train")
        valid_dataset = NpzDataset(file_path, operation="valid")
        test_dataset = NpzDataset(file_path, operation="test")

        train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset), shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

        return train_loader, valid_loader, test_loader

    @staticmethod
    def sparse_elm_autoencoder(a: np.ndarray, b: np.ndarray, lam: float, itrs: int) -> np.ndarray:
        """

        param a: a matrix with size (d, n), where d is the dimension of the input fcnn_data and n is the number of
        training samples.
        param b: a matrix with size (d, m), where m is the number of hidden neurons in the autoencoder.
        param lam: a scalar that controls the sparsity of the learned representation.
        param itrs: the number of iterations for training the autoencoder.
        :return: The function returns a matrix "x" with size (m, n), which is the learned representation of the input
                 fcnn_data.
        """

        # These lines calculate the Lipschitz constant of the input matrix "a", which is used to set the step size of
        # the FISTA algorithm. The matrix multiplication "a.T @ a" computes the inner product of "a" with itself, and
        # the np.linalg.eigvals function computes the eigenvalues of this matrix. The largest eigenvalue is then used to
        # set the Lipschitz constant "li".
        aa = a.T @ a
        lf = np.linalg.eigvals(aa)
        lf = np.real(lf)
        lf = np.max(lf)
        li = 1 / lf

        # This line sets the regularization parameter "alp" based on the Lipschitz constant li and a user-specified
        # value "lam".
        alp = lam * li

        # Initialize the weights and other variables
        m = a.shape[1]
        n = b.shape[1]
        x = np.zeros((m, n))  # weight matrix
        yk = x
        tk = 1  # step size
        l1 = 2 * li * aa
        l2 = 2 * li * a.T @ b

        # Perform a specified number of iterations of the FISTA algorithm to learn the weights
        for _ in range(itrs):
            # Compute the next estimate of the weights
            ck = yk - l1 @ yk + l2
            x1 = np.multiply(np.sign(ck), np.maximum(np.abs(ck) - alp, 0))  # updated weight matrix

            # Update the momentum parameter
            tk1 = 0.5 + 0.5 * np.sqrt(1 + 4 * tk ** 2)

            # Compute the next estimate of the momentum
            tt = (tk - 1) / tk1
            yk = x1 + tt * (x - x1)
            tk = tk1
            x = x1

        # Return the learned weights
        return x

    @staticmethod
    def min_max_scale(matrix: np.ndarray, scale: str) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        The function scales the values of the matrix to either the range of [-1, 1] or [0, 1] based on the value of
        the parameter "scale". The function returns the scaled fcnn_data along with a list that contains the minimum
        values, maximum values, and ranges for each row.

        :param matrix: Input matrix to be scaled.
        :param scale: String specifying the scaling range ("-1_1" or "0_1").
        :return: Tuple containing the scaled matrix and a list with minimum values, maximum values, and ranges for each
        row.
        """

        min_vals = np.min(matrix, axis=1).reshape(-1, 1)
        max_vals = np.max(matrix, axis=1).reshape(-1, 1)
        ranges = max_vals - min_vals
        if scale == "-1_1":
            data = 2 * (matrix - min_vals) / ranges - 1
            return data, [min_vals, max_vals, ranges]
        elif scale == "0_1":
            data = (matrix - min_vals) / ranges
            return data, [min_vals, max_vals, ranges]
        else:
            raise ValueError("Wrong value!")
    
    @staticmethod
    def apply_normalization(data, min_values, range_values):
        """
        The purpose of this function is to normalize the input fcnn_data based on the given minimum values and range
        values.

        :param data: Input data to be normalized.
        :param min_values: Minimum values used for normalization.
        :param range_values: Range values used for normalization.
        :return: Normalized data.
        """

        return (data - min_values) / range_values

    @measure_execution_time
    def train(self):
        """
        Train the model.

        :return: A tuple containing the results of training.
        """

        random_weights_1 = (
                2 * np.random.rand(self.num_features + 1, self.cfg.get("hidden_neurons").get(self.cfg.get("dataset_name"))[0]) - 1
        )
        random_weights_2 = (
                2 * np.random.rand(self.cfg.get("hidden_neurons").get(self.cfg.get("dataset_name"))[0] + 1,
                                   self.cfg.get("hidden_neurons").get(self.cfg.get("dataset_name"))[1]) - 1
        )

        self.random_weights_3 = (
            (2 * np.random.rand(self.cfg.get("hidden_neurons").get(self.cfg.get("dataset_name"))[1] + 1,
                                self.cfg.get("hidden_neurons").get(self.cfg.get("dataset_name"))[2]) - 1).T
        )
        self.random_weights_3 = linalg.orth(self.random_weights_3).T

        for train_data, train_labels in self.train_loader:
            # First layer RELM
            h1 = np.hstack([train_data, np.ones((train_data.shape[0], 1)) * 0.1])
            a1 = h1 @ random_weights_1
            a1, _ = self.min_max_scale(a1, "-1_1")
            beta1 = self.sparse_elm_autoencoder(a=a1, b=h1, lam=1e-3, itrs=50)
            del a1
            t1 = h1 @ beta1.T
            del h1
            t1, ps1 = self.min_max_scale(t1.T, "0_1")

            # Second layer RELM
            h2 = np.hstack([t1.T, np.ones((t1.T.shape[0], 1)) * 0.1])
            del t1
            a2 = h2 @ random_weights_2
            a2, _ = self.min_max_scale(a2, "-1_1")
            beta2 = self.sparse_elm_autoencoder(a=a2, b=h2, lam=1e-3, itrs=50)
            del a2
            t2 = h2 @ beta2.T
            del h2
            t2, ps2 = self.min_max_scale(t2.T, "0_1")

            # Original ELM
            h3 = np.hstack([t2.T, np.ones((t2.T.shape[0], 1)) * 0.1])
            del t2
            t3 = h3 @ self.random_weights_3
            del h3
            l3 = np.amax(np.amax(t3))
            l3 = self.scaling_factor / l3

            t3 = np.tanh(t3 * l3)
            # Finsh Training
            beta = np.linalg.solve(t3.T.dot(t3) + np.eye(t3.shape[1]) * self.C_penalty, t3.T.dot(train_labels))

            return t3, beta, beta1, beta2, l3, ps1, ps2

    def training_accuracy(self, t3: np.ndarray, beta: np.ndarray) -> list:
        """
        Calculate and log the training accuracy.

        :param t3: Output from the last layer.
        :param beta: Weight matrix.
        """

        for _, train_labels in self.train_loader:
            y_predicted = t3 @ beta
            y_predicted_argmax = np.argmax(np.asarray(y_predicted), axis=-1)
            y_true_argmax = np.argmax(np.asarray(train_labels), axis=-1)
            accuracy = accuracy_score(y_true_argmax, y_predicted_argmax)
            precision = precision_score(y_true_argmax, y_predicted_argmax, average='macro', zero_division=0)
            recall = recall_score(y_true_argmax, y_predicted_argmax, average='macro', zero_division=0)
            f1sore = f1_score(y_true_argmax, y_predicted_argmax, average='macro', zero_division=0)
            training_time = self.train.execution_time

            logging.info(f"Training Accuracy is: {accuracy:.4f}%")
            logging.info(f"Training precision is: {precision:.4f}%")
            logging.info(f"Training recall is: {recall:.4f}%")
            logging.info(f"Training f1sore is: {f1sore:.4f}%")
            logging.info(f"Training time is: {training_time:.4f}%")

            return [accuracy, precision, recall, f1sore, training_time]

    def testing_accuracy(self,
                         beta: np.ndarray,
                         beta1: np.ndarray,
                         beta2: np.ndarray,
                         l3: float,
                         ps1: list,
                         ps2: list):
        """
        Calculate and log the testing accuracy.

        :param beta: Weight matrix.
        :param beta1: Weight matrix for the first layer.
        :param beta2: Weight matrix for the second layer.
        :param l3: Scaling factor for the third layer.
        :param ps1: Tuple containing minimum values, maximum values, and ranges for the first layer.
        :param ps2: Tuple containing minimum values, maximum values, and ranges for the second layer.
        """

        for test_data, test_labels in self.test_loader:
            #  First layer feedforward
            hh1 = np.hstack([test_data, np.ones((test_data.shape[0], 1)) * 0.1])
            tt1 = hh1 @ beta1.T
            tt1 = self.apply_normalization(tt1.T, ps1[0], ps1[2])
            tt1 = tt1.T

            # # Second layer feedforward
            hh2 = np.hstack([tt1, np.ones((tt1.shape[0], 1)) * 0.1])
            tt2 = hh2 @ beta2.T
            tt2 = self.apply_normalization(tt2.T, ps2[0], ps2[2])
            tt2 = tt2.T

            # Last layer feedforward
            hh3 = np.hstack([tt2, np.ones((tt2.shape[0], 1)) * 0.1])
            tt3 = np.tanh(hh3 @ self.random_weights_3 * l3)

            y_predicted = tt3 @ beta
            del tt3

            y_predicted_argmax = np.argmax(np.asarray(y_predicted), axis=-1)
            y_true_argmax = np.argmax(np.asarray(test_labels), axis=-1)

            accuracy = accuracy_score(y_true_argmax, y_predicted_argmax)
            precision = precision_score(y_true_argmax, y_predicted_argmax, average='macro', zero_division=0)
            recall = recall_score(y_true_argmax, y_predicted_argmax, average='macro', zero_division=0)
            f1sore = f1_score(y_true_argmax, y_predicted_argmax, average='macro', zero_division=0)

            logging.info(f"Testing Accuracy is: {accuracy:.4f}%")
            logging.info(f"Testing precision is: {precision:.4f}%")
            logging.info(f"Testing recall is: {recall:.4f}%")
            logging.info(f"Testing f1sore is: {f1sore:.4f}%")

            return [accuracy, precision, recall, f1sore]

    def main(self) -> None:
        """
         Execute the main logic of the program.
         """

        for idx in tqdm(range(self.cfg.get("num_tests")), desc="Evaluation"):
            t3, beta, beta1, beta2, l3, ps1, ps2 = self.train()
            training_metrics = self.training_accuracy(t3, beta)
            testing_metrics = self.testing_accuracy(beta, beta1, beta2, l3, ps1, ps2)

            metrics = reorder_metrics_lists(train_metrics=training_metrics,
                                            test_metrics=testing_metrics)
            insert_data_to_excel(self.filename, self.cfg.get("dataset_name"), idx + 2, metrics, "helm")

        average_columns_in_excel(self.filename)


if __name__ == "__main__":
    for i in range(1):
        helm = HELM()
        helm.main()
