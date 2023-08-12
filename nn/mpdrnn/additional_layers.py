import numpy as np

from elm.src.utils.activation_functions import leaky_ReLU, identity, ReLU, sigmoid, tanh
from elm.src.utils.loss_functions import mae, mse


class AdditionalLayer(object):
    def __init__(self, previous_layer, n_hidden_nodes: int = 256, n_output_nodes: int = 10, mu: float = 0,
                 sigma: float = 10, activation: str = "ReLU", loss: str = "mse", name: str = None,
                 method: str = "BASE"):

        # Initialize attributes
        self.H_prev = previous_layer.weights["H_prev"]
        self.previous_weights = previous_layer.weights["prev_weights"]
        self.predict_h_train_prev = previous_layer.weights["predict_h_train"]
        self.predict_h_test_prev = previous_layer.weights["predict_h_test"]

        self.method = method
        self.name = name
        self.__n_hidden_nodes = n_hidden_nodes
        self.__n_output_nodes = n_output_nodes

        self.H_i_layer = None
        self.H_i_pseudo_inverse = None
        self.predict_h_train = None
        self.predict_h_test = None
        self.new_weights = None

        self.hidden_layer_i = self.__create_hidden_layer_i(mu, sigma)

        # Set activation and loss functions
        self.__init_mapping_dicts()
        self.__activation = self.activation_map.get(activation, None)
        self.__loss = self.loss_map.get(loss, None)

    def __init_mapping_dicts(self):
        """

        :return:
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

    def __create_hidden_layer_i(self, mu, sigma):
        """

        :param mu:
        :param sigma:
        :return:
        """

        noise = np.random.normal(mu, sigma, (self.previous_weights.shape[0],
                                             self.previous_weights.shape[1]))
        w_rnd_out_i = self.previous_weights + noise
        hidden_layer_i_a = np.hstack((self.previous_weights, w_rnd_out_i))

        if self.method in ["BASE"]:
            w_rnd = np.random.normal(mu, sigma, (self.previous_weights.shape[0],
                                                 self.__n_hidden_nodes))

            hidden_layer_i = np.hstack((hidden_layer_i_a, w_rnd))
        else:
            w_rnd = np.random.normal(mu, sigma, (self.previous_weights.shape[0], self.__n_hidden_nodes // 2))
            ort = (np.apply_along_axis(self.calc_ort, axis=1, arr=w_rnd))
            ort = np.squeeze(ort)
            w_rnd = np.hstack((w_rnd, ort))
            hidden_layer_i = np.hstack((hidden_layer_i_a, w_rnd))

        return hidden_layer_i

    def calc_ort(self, x):
        """
        # Calculation of orthogonal neuron

        :param x:
        :return:
        """

        return np.subtract(x, np.dot(self.predict_h_train_prev, np.dot(np.transpose(self.predict_h_train_prev), x)))

    def fit(self, labels: np.ndarray):
        """

        param labels:
        :return:
        """

        self.H_i_layer = self.__activation(self.H_prev @ self.hidden_layer_i)
        self.H_i_pseudo_inverse = np.linalg.pinv(self.H_i_layer)
        self.new_weights = self.H_i_pseudo_inverse @ labels

    def predict(self, operation: str):
        """

        :param operation:
        :return:
        """

        if operation not in ["train", "test"]:
            raise ValueError('An unknown operation \'%s\'.' % operation)

        if operation == "train":
            self.predict_h_train = self.__activation(self.predict_h_train_prev @ self.hidden_layer_i)
            return self.predict_h_train @ self.new_weights
        else:
            self.predict_h_test = self.__activation(self.predict_h_test_prev @ self.hidden_layer_i)
            return self.predict_h_test @ self.new_weights

    def evaluate(self, labels: np.ndarray, metrics: dict = None, operation: str = "train") -> dict:
        """

        :param labels:
        :param metrics:
        :param operation:
        :return:
        """

        if metrics is None:
            raise ValueError("No metric is given!")

        y_pred = self.predict(operation)
        y_predicted_argmax = np.argmax(np.asarray(y_pred), axis=-1)
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

        :return:
        """

        return {
            'prev_weights': self.new_weights,
            'hidden_i_prev': self.hidden_layer_i,
            'H_prev': self.H_i_layer,
            'predict_h_train': self.predict_h_train,
            'predict_h_test': self.predict_h_test
        }
