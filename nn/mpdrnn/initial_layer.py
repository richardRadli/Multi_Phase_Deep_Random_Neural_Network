import numpy as np

from numpy.linalg import pinv

from config.config import MPDRNNConfig
from utils.activation_functions import leaky_ReLU, identity, ReLU, sigmoid, tanh
from utils.biases import ones_bias, uniform_bias, xavier_bias, zero_bias
from utils.loss_functions import mae, mse


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++ I N I T I A L   L A Y E R ++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class InitialLayer(object):
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- __I N I T__ ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 train_data: np.ndarray,
                 train_labels: np.ndarray,
                 test_data: np.ndarray,
                 test_labels: np.ndarray,
                 n_input_nodes: int,
                 n_hidden_nodes: int,
                 activation: str = "ReLU",
                 loss: str = "mse",
                 name: str = None,
                 method: str = "BASE",
                 bias: str = "zero"):
        """
        Initialize an InitialLayer instance.

        :param train_data: Training fcnn_data.
        :param train_labels: Training labels.
        :param test_data: Test fcnn_data.
        :param test_labels: Test labels.
        :param n_input_nodes: Number of input nodes.
        :param n_hidden_nodes: Number of hidden nodes.
        :param activation: Name of the activation function.
        :param loss: Name of the loss function.
        :param name: Name of the layer.
        :param method: Training method.
        :param bias: Bias initialization method.
        """

        self.cfg = MPDRNNConfig().parse()

        self.name = name
        self.method = method

        # Variables for the first phase
        self.H1 = None
        self.H_pseudo_inverse = None
        self.__n_input_nodes = n_input_nodes
        self.__n_hidden_nodes = n_hidden_nodes
        self.__alpha_weights = np.random.uniform(low=self.cfg.alpha_weights_min,
                                                 high=self.cfg.alpha_weights_max,
                                                 size=(self.__n_input_nodes, self.__n_hidden_nodes))
        self.__beta_weights = None

        # Variables to store prediction results
        self.predict_h_train = None
        self.predict_h_test = None

        self.__init_mapping_dicts()
        self.__activation = self.activation_map.get(activation, None)
        self.__loss = self.loss_map.get(loss, None)
        self.__bias = self.bias_map.get(bias, None)

        # Load the fcnn_data directly
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------- I N I T   M A P P I N G   D I C T S --------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init_mapping_dicts(self) -> None:
        """
        Initializes mapping dictionaries for activation functions, loss functions, and bias initialization methods.

        :return: None
        """

        self.activation_map = {
            "sigmoid": sigmoid,
            "identity": identity,
            "ReLU": ReLU,
            "leaky_ReLU": leaky_ReLU,
            "tanh": tanh
        }
        self.loss_map = {
            "mse": mse,
            "mae": mae
        }
        self.bias_map = {
            "zero": zero_bias,
            "xavier": xavier_bias,
            "uniform": uniform_bias,
            "ones": ones_bias
        }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- T R A I N ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def train(self) -> None:
        """
        Trains the initial layer of the ELM model.

        :return: None
        """

        # Compute the first hidden layer (size: [number of fcnn_data, number of hidden nodes])
        self.H1 = self.__activation(self.train_data @ self.__alpha_weights)

        if self.method in ["BASE", "EXP_ORT"]:
            # Compute inverse of H1 (size: (number of hidden nodes, number of fcnn_data))
            self.H_pseudo_inverse = pinv(self.H1)

            # Compute the beta weights (size: [number of hidden nodes, number of output nodes])
            self.__beta_weights = self.H_pseudo_inverse @ self.train_labels
        else:
            C = 0.01
            identity_mtx = np.identity(self.H1.shape[1])
            if self.H1.shape[0] > self.H1.shape[1]:
                self.__beta_weights = (
                        np.linalg.pinv(self.H1.T @ self.H1 + identity_mtx / C) @ self.H1.T @ self.train_labels)
            elif self.H1.shape[0] < self.H1.shape[1]:
                self.__beta_weights = (
                        self.H1.T @ np.linalg.pinv(self.H1 @ self.H1.T + identity_mtx / C) @ self.train_labels)
            else:
                raise ValueError("not possible")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- P R E D I C T --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def predict(self, data: np.ndarray, operation: str) -> np.ndarray:
        """
        Predict the model on train/test fcnn_data.

        :param data: Input fcnn_data.
        :param operation: Either train or test.
        :return: Predicted values.
        """

        if operation not in ["train", "test"]:
            raise ValueError('An unknown operation \'%s\'.' % operation)

        if operation == "train":
            predicted_hidden = self.__activation(np.dot(data, self.__alpha_weights))
            self.predict_h_train = predicted_hidden
            return np.dot(predicted_hidden, self.__beta_weights)
        else:
            predicted_hidden = self.__activation(np.dot(data, self.__alpha_weights))
            self.predict_h_test = predicted_hidden
            return np.dot(predicted_hidden, self.__beta_weights)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- E V A L U A T E ------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def evaluate(self, data: np.ndarray, labels: np.ndarray, metrics: dict = None, operation: str = "train") -> dict:
        """
        Evaluate the model's performance using the specified metrics.

        :param data: Input fcnn_data for evaluation.
        :param labels: True labels for the input fcnn_data.
        :param metrics: Dictionary of metric names and corresponding metric functions.
        :param operation: Either train or test.
        :return: Dictionary containing evaluated metrics.
        """

        if metrics is None:
            raise ValueError("No metrics are given!")

        y_predicted = self.predict(data, operation)
        y_predicted_argmax = np.argmax(np.asarray(y_predicted), axis=-1)
        y_true_argmax = np.argmax(np.asarray(labels), axis=-1)

        evaluated_metrics = {}

        for metric_name, metric_function in metrics.items():
            if callable(metric_function):
                metric_value = metric_function(y_true_argmax, y_predicted_argmax)
                evaluated_metrics[metric_name] = metric_value
            else:
                raise ValueError("Invalid metric function provided for %s." % metric_name)

        return evaluated_metrics

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- W E I G H T S -------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @property
    def weights(self):
        """
        Function to save weights.

        :return: Values of the weights.
        """

        return {
            'prev_weights': self.__beta_weights,
            'H_prev': self.H1,
            'predict_h_train': self.predict_h_train,
            'predict_h_test': self.predict_h_test
        }
