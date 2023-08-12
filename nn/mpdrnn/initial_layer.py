import numpy as np

from numpy.linalg import pinv

from elm.src.config.config import MPDRNNConfig
from elm.src.utils.activation_functions import leaky_ReLU, identity, ReLU, sigmoid, tanh
from elm.src.utils.biases import ones_bias, uniform_bias, xavier_bias, zero_bias
from elm.src.utils.loss_functions import mae, mse


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++ I N I T I A L   L A Y E R ++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class InitialLayer(object):
    def __init__(self, train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray, test_labels: np.ndarray,
                 n_input_nodes: int, n_hidden_nodes: int, activation: str = "ReLU", loss: str = "mse", name: str = None,
                 method: str = "BASE", bias: str = "zero"):

        self.cfg = MPDRNNConfig().parse()

        self.name = name
        self.method = method

        # Variables for the first phase
        self.H1 = None
        self.H_pseudo_inverse = None
        self.__n_input_nodes = n_input_nodes
        self.__n_hidden_nodes = n_hidden_nodes
        self.__alpha_weights = None
        self.__beta_weights = None

        # Variables to store prediction results
        self.predict_h_train = None
        self.predict_h_test = None

        self.__init_mapping_dicts()
        self.__activation = self.activation_map.get(activation, None)
        self.__loss = self.loss_map.get(loss, None)
        self.__bias = self.bias_map.get(bias, None)

        # Load the data directly
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------- I N I T   M A P P I N G   D I C T S --------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init_mapping_dicts(self):
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
    # ------------------------------------------------------ F I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def fit(self):
        if self.method in ["BASE", "EXP"]:
            self.__alpha_weights = np.random.uniform(low=self.cfg.alpha_weights_min, high=self.cfg.alpha_weights_max,
                                                     size=(self.__n_input_nodes, self.__n_hidden_nodes))
        else:
            raise ValueError("Wrong method was given!")

        # Compute the first hidden layer (size: [number of data, number of hidden nodes])
        self.H1 = self.__activation(self.train_data @ self.__alpha_weights)

        # Compute inverse of H1 (size: (number of hidden nodes, number of data))
        self.H_pseudo_inverse = pinv(self.H1)

        # Compute the beta weights (size: [number of hidden nodes, number of output nodes])
        self.__beta_weights = self.H_pseudo_inverse @ self.train_labels

    def predict(self, data: np.ndarray, operation: str) -> np.ndarray:
        """
        Function to predict the model on train/test data.
        param data: Input data.
        param operation: Either train or test.
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

    def evaluate(self, data: np.ndarray, labels: np.ndarray, metrics: dict = None, operation: str = "train") -> dict:
        """

        :param data:
        :param labels:
        :param metrics:
        :param operation:
        :return:
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
