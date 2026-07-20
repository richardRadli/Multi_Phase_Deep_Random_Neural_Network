import colorama
import logging
import numpy as np
import scipy.linalg as linalg

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import List, Tuple

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import general_dataset_configs
from utils.utils import load_config_json, setup_logger, measure_execution_time, create_train_test_datasets


class HELMBase:
    def __init__(self, override_cfg: dict = None, celery_task=None):
        setup_logger()
        self.celery_task = celery_task

        if override_cfg is not None:
            self.cfg = override_cfg
        else:
            self.cfg = (
                load_config_json(json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_helm"),
                                 json_filename=JSON_FILES_PATHS.get_data_path("config_helm"))
            )

        if self.cfg.get("seed"):
            np.random.seed(self.cfg.get("seed"))

        file_path = general_dataset_configs(self.cfg.get("dataset_name")).get("cached_dataset_file")
        self.train_loader, self.test_loader = (
            create_train_test_datasets(file_path)
        )

        self.num_features = general_dataset_configs(self.cfg.get("dataset_name")).get("num_features")
        self.class_labels = general_dataset_configs(self.cfg.get("dataset_name")).get("class_labels", None)

        self.random_weights_3 = (
            (2 * np.random.rand(self.cfg.get("hidden_neurons")[1] + 1,
                                self.cfg.get("hidden_neurons")[2]) - 1).T
        )

        self.train_cm_list = []
        self.test_cm_list = []

        colorama.init()

    @staticmethod
    def sparse_elm_autoencoder(a: np.ndarray, b: np.ndarray, lam: float, itrs: int) -> np.ndarray:
        aa = a.T @ a
        lf = np.linalg.eigvals(aa)
        lf = np.real(lf)
        lf = np.max(lf)
        li = 1 / lf

        alp = lam * li

        m = a.shape[1]
        n = b.shape[1]
        x = np.zeros((m, n))
        yk = x
        tk = 1
        l1 = 2 * li * aa
        l2 = 2 * li * a.T @ b

        for _ in range(itrs):
            ck = yk - l1 @ yk + l2
            x1 = np.multiply(np.sign(ck), np.maximum(np.abs(ck) - alp, 0))

            tk1 = 0.5 + 0.5 * np.sqrt(1 + 4 * tk ** 2)
            tt = (tk - 1) / tk1
            yk = x1 + tt * (x - x1)
            tk = tk1
            x = x1

        return x

    @staticmethod
    def min_max_scale(matrix: np.ndarray, scale: str) -> Tuple[np.ndarray, List[np.ndarray]]:
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
    def apply_normalization(data: np.ndarray, min_values: np.ndarray, range_values: np.ndarray) -> np.ndarray:
        return (data - min_values) / range_values

    @measure_execution_time
    def train(self, config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, list, list]:
        random_weights_1 = (
                2 * np.random.rand(self.num_features + 1,
                                   self.cfg.get("hidden_neurons")[0]) - 1
        )
        random_weights_2 = (
                2 * np.random.rand(self.cfg.get("hidden_neurons")[0] + 1,
                                   self.cfg.get("hidden_neurons")[1]) - 1
        )

        self.random_weights_3 = linalg.orth(self.random_weights_3).T

        for train_data, train_labels in self.train_loader:
            num_classes = train_labels.shape[1]

            if hasattr(self, "class_labels") and self.class_labels is not None:
                self.labels = self.class_labels
            else:
                self.labels = [str(i) for i in range(num_classes)]


            # First layer RELM
            h1 = np.hstack([train_data, np.ones((train_data.shape[0], 1)) * 0.1])
            a1 = h1 @ random_weights_1
            a1, _ = self.min_max_scale(a1, "-1_1")
            beta1 = self.sparse_elm_autoencoder(a=a1, b=h1, lam=1e-3, itrs=50)
            del a1
            t1 = h1 @ beta1.T
            del h1
            t1, ps1 = self.min_max_scale(t1.T, "0_1")

            self.t1_train = t1.T

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

            self.t2_train = t2.T

            # Original ELM
            h3 = np.hstack([t2.T, np.ones((t2.T.shape[0], 1)) * 0.1])
            del t2
            t3 = h3 @ self.random_weights_3
            del h3
            l3 = np.amax(np.amax(t3))
            l3 = config.get("scaling_factor") / l3

            t3 = np.tanh(t3 * l3)
            # Finish Training
            beta = np.linalg.solve(t3.T.dot(t3) + np.eye(t3.shape[1]) * config.get("C_penalty"), t3.T.dot(train_labels))

            return t3, beta, beta1, beta2, l3, ps1, ps2

    def training_accuracy(self, t3: np.ndarray, beta: np.ndarray) -> list:
        for _, train_labels in self.train_loader:
            penalty = self.cfg.get("penalty") or 1e-3
            y_true_argmax = np.argmax(np.asarray(train_labels), axis=-1)

            self.beta_l1 = np.linalg.solve(
                self.t1_train.T.dot(self.t1_train) + np.eye(self.t1_train.shape[1]) * penalty,
                self.t1_train.T.dot(train_labels)
            )
            y_pred_l1 = self.t1_train @ self.beta_l1
            y_pred_l1_argmax = np.argmax(np.asarray(y_pred_l1), axis=-1)
            cm_l1 = confusion_matrix(y_true_argmax, y_pred_l1_argmax)

            self.beta_l2 = np.linalg.solve(
                self.t2_train.T.dot(self.t2_train) + np.eye(self.t2_train.shape[1]) * penalty,
                self.t2_train.T.dot(train_labels)
            )
            y_pred_l2 = self.t2_train @ self.beta_l2
            y_pred_l2_argmax = np.argmax(np.asarray(y_pred_l2), axis=-1)
            cm_l2 = confusion_matrix(y_true_argmax, y_pred_l2_argmax)

            y_predicted = t3 @ beta
            y_predicted_argmax = np.argmax(np.asarray(y_predicted), axis=-1)
            cm_l3 = confusion_matrix(y_true_argmax, y_predicted_argmax)

            self.train_cm_list = [cm_l1, cm_l2, cm_l3]

            accuracy = accuracy_score(y_true_argmax, y_predicted_argmax)
            precision = precision_score(y_true_argmax, y_predicted_argmax, average='macro', zero_division=0)
            recall = recall_score(y_true_argmax, y_predicted_argmax, average='macro', zero_division=0)
            f1sore = f1_score(y_true_argmax, y_predicted_argmax, average='macro', zero_division=0)
            training_time = self.train.execution_time

            logging.info(f"Training Accuracy is: {accuracy:.4f}%")
            logging.info(f"Training precision is: {precision:.4f}")
            logging.info(f"Training recall is: {recall:.4f}")
            logging.info(f"Training f1sore is: {f1sore:.4f}")
            logging.info(f"Training time is: {training_time:.4f} seconds")

            return [accuracy, precision, recall, f1sore, training_time]

    def evaluation(self, beta: np.ndarray, beta1: np.ndarray, beta2: np.ndarray, l3: float, ps1: list, ps2: list,
                   dataloader) -> list:
        for data, labels in dataloader:
            y_true_argmax = np.argmax(np.asarray(labels), axis=-1)

            #  First layer feedforward
            hh1 = np.hstack([data, np.ones((data.shape[0], 1)) * 0.1])
            tt1 = hh1 @ beta1.T
            tt1 = self.apply_normalization(tt1.T, ps1[0], ps1[2])
            tt1 = tt1.T

            # 1. Réteg teszt konfúziós mátrix
            y_pred_l1 = tt1 @ self.beta_l1
            y_pred_l1_argmax = np.argmax(np.asarray(y_pred_l1), axis=-1)
            cm_l1 = confusion_matrix(y_true_argmax, y_pred_l1_argmax)

            # Second layer feedforward
            hh2 = np.hstack([tt1, np.ones((tt1.shape[0], 1)) * 0.1])
            tt2 = hh2 @ beta2.T
            tt2 = self.apply_normalization(tt2.T, ps2[0], ps2[2])
            tt2 = tt2.T

            # 2. Réteg teszt konfúziós mátrix
            y_pred_l2 = tt2 @ self.beta_l2
            y_pred_l2_argmax = np.argmax(np.asarray(y_pred_l2), axis=-1)
            cm_l2 = confusion_matrix(y_true_argmax, y_pred_l2_argmax)

            # Last layer feedforward
            hh3 = np.hstack([tt2, np.ones((tt2.shape[0], 1)) * 0.1])
            tt3 = np.tanh(hh3 @ self.random_weights_3 * l3)

            # 3. Réteg teszt konfúziós mátrix
            y_predicted = tt3 @ beta
            del tt3
            y_predicted_argmax = np.argmax(np.asarray(y_predicted), axis=-1)
            cm_l3 = confusion_matrix(y_true_argmax, y_predicted_argmax)

            self.test_cm_list = [cm_l1, cm_l2, cm_l3]

            accuracy = accuracy_score(y_true_argmax, y_predicted_argmax)
            precision = precision_score(y_true_argmax, y_predicted_argmax, average='macro', zero_division=0)
            recall = recall_score(y_true_argmax, y_predicted_argmax, average='macro', zero_division=0)
            f1sore = f1_score(y_true_argmax, y_predicted_argmax, average='macro', zero_division=0)

            logging.info(f"Testing Accuracy is: {accuracy:.4f}%")
            logging.info(f"Testing precision is: {precision:.4f}")
            logging.info(f"Testing recall is: {recall:.4f}")
            logging.info(f"Testing f1sore is: {f1sore:.4f}")

            return [accuracy, precision, recall, f1sore]