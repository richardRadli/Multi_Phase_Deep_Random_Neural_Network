from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, precision_recall_fscore_support

from additional_layers import AdditionalLayer
from elm.src.config.config import MPDRNNConfig
from elm.src.config.dataset_config import general_dataset_configs
from initial_layer import InitialLayer
from elm.src.utils.utils import pretty_print_results, measure_execution_time


class ELMModelWrapper:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        # Initialize config and paths
        self.cfg = MPDRNNConfig().parse()
        gen_ds_cfg = general_dataset_configs(self.cfg)

        # Initialize data
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

        # Initialize number of neurons
        self.n_output_nodes = gen_ds_cfg.get("num_classes")
        n_hidden_nodes_layer_1 = (
            gen_ds_cfg.get("eq_neurons"))[0] if self.cfg.method == "BASE" else gen_ds_cfg.get("exp_neurons")[0]
        self.n_hidden_nodes_layer_2 = (
            gen_ds_cfg.get("eq_neurons"))[1] if self.cfg.method == "BASE" else gen_ds_cfg.get("exp_neurons")[1]
        self.n_hidden_nodes_layer_3 = (
            gen_ds_cfg.get("eq_neurons"))[2] if self.cfg.method == "BASE" else gen_ds_cfg.get("exp_neurons")[2]

        # Initialize layers
        self.initial_layer = InitialLayer(train_data=train_data, train_labels=train_labels, test_data=test_data,
                                          test_labels=test_labels, n_input_nodes=gen_ds_cfg.get("num_features"),
                                          n_hidden_nodes=n_hidden_nodes_layer_1, activation="ReLU", loss="mse",
                                          name="phase_1", method=self.cfg.method, bias="zero")
        self.second_layer = None
        self.third_layer = None

        self.metrics = {
            "accuracy": accuracy_score,
            "precision_recall_fscore": lambda y_true, y_pred: precision_recall_fscore_support(y_true, y_pred,
                                                                                              average="macro"),
            "confusion_matrix": lambda y_true, y_pred: confusion_matrix(y_true, y_pred),
            "loss": lambda y_true, y_pred: mean_squared_error(y_true, y_pred)
        }

        self.total_execution_time = {}

    @measure_execution_time
    def train_first_layer(self):
        self.initial_layer.fit()

    def evaluate_first_layer(self):
        train_metrics = self.initial_layer.evaluate(data=self.train_data, labels=self.train_labels,
                                                    metrics=self.metrics, operation="train")
        test_metrics = self.initial_layer.evaluate(data=self.test_data, labels=self.test_labels,
                                                   metrics=self.metrics, operation="test")

        pretty_print_results(train_metrics, operation="train", name=self.initial_layer.name,
                             training_time=self.total_execution_time.get("train_first_layer"))
        pretty_print_results(test_metrics, name=self.initial_layer.name, operation="test")

    def init_second_layer(self):
        return AdditionalLayer(self.initial_layer, n_hidden_nodes=self.n_hidden_nodes_layer_2,
                               n_output_nodes=self.n_output_nodes, mu=0, sigma=10, activation="ReLU",
                               loss="mse", name="phase_2", method=self.cfg.method)

    @measure_execution_time
    def train_second_layer(self):
        self.second_layer = self.init_second_layer()
        self.second_layer.fit(self.train_labels)

    def evaluate_second_layer(self):
        train_metrics = self.second_layer.evaluate(labels=self.train_labels, metrics=self.metrics, operation="train")
        test_metrics = self.second_layer.evaluate(labels=self.test_labels, metrics=self.metrics, operation="test")
        pretty_print_results(metric_results=train_metrics, operation="train", name=self.second_layer.name,
                             training_time=self.total_execution_time.get("train_second_layer"))
        pretty_print_results(metric_results=test_metrics, name=self.second_layer.name, operation="test")

    def init_third_layer(self):
        return AdditionalLayer(self.initial_layer, n_hidden_nodes=self.n_hidden_nodes_layer_3,
                               n_output_nodes=self.n_output_nodes, mu=0, sigma=10, activation="ReLU",
                               loss="mse", name="phase_3", method=self.cfg.method)

    @measure_execution_time
    def train_third_layer(self):
        self.third_layer = self.init_third_layer()
        self.third_layer.fit(self.train_labels)

    def evaluate_third_layer(self):
        train_metrics = self.third_layer.evaluate(labels=self.train_labels, metrics=self.metrics, operation="train")
        test_metrics = self.third_layer.evaluate(labels=self.test_labels, metrics=self.metrics, operation="test")
        pretty_print_results(metric_results=train_metrics, operation="train", name=self.third_layer.name,
                             training_time=self.total_execution_time.get("train_third_layer"))
        pretty_print_results(metric_results=test_metrics, name=self.third_layer.name, operation="test")

    def get_total_execution_time(self):
        total_time = sum(self.total_execution_time.values())
        return total_time
