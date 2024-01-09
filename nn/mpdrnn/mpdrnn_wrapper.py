import os

from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, precision_recall_fscore_support

from additional_layers import AdditionalLayer
from config.config import MPDRNNConfig
from config.dataset_config import general_dataset_configs
from utils.utils import create_timestamp, measure_execution_time, pretty_print_results, plot_confusion_matrix
from initial_layer import InitialLayer


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++ E L M   M O D E L   W R A P P E R ++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ELMModelWrapper:
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- __I N I T__ ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, train_data, train_labels, test_data, test_labels):
        # Initialize config and paths
        self.cfg = MPDRNNConfig().parse()
        self.gen_ds_cfg = general_dataset_configs(self.cfg)

        # Initialize fcnn_data
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

        self.training_accuracy = []
        self.testing_accuracy = []

        # Initialize number of neurons
        self.n_output_nodes = self.gen_ds_cfg.get("num_classes")
        self.n_hidden_nodes_layer_1 = (
            self.gen_ds_cfg.get("eq_neurons"))[0] if self.cfg.method == "BASE" \
            else self.gen_ds_cfg.get("exp_neurons")[0]
        self.n_hidden_nodes_layer_2 = (
            self.gen_ds_cfg.get("eq_neurons"))[1] if self.cfg.method == "BASE" \
            else self.gen_ds_cfg.get("exp_neurons")[1]
        self.n_hidden_nodes_layer_3 = (
            self.gen_ds_cfg.get("eq_neurons"))[2] if self.cfg.method == "BASE" \
            else self.gen_ds_cfg.get("exp_neurons")[2]

        # Initialize layers
        self.initial_layer = None
        self.second_layer = None
        self.third_layer = None

        # Define metrics
        self.metrics = {
            "accuracy": accuracy_score,
            "precision_recall_fscore": lambda y_true, y_pred: precision_recall_fscore_support(y_true, y_pred,
                                                                                              average="macro"),
            "confusion_matrix": lambda y_true, y_pred: confusion_matrix(y_true, y_pred),
            "loss": lambda y_true, y_pred: mean_squared_error(y_true, y_pred)
        }

        self.total_execution_time = {}

        timestamp = create_timestamp()
        self.path_to_cm = os.path.join(self.gen_ds_cfg.get("path_to_cm"), timestamp)
        os.makedirs(self.path_to_cm, exist_ok=True)
        self.path_to_metrics_plot = os.path.join(self.gen_ds_cfg.get("path_to_metrics_plot"), timestamp)
        os.makedirs(self.path_to_metrics_plot, exist_ok=True)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- I N I T   F I R S T   L A Y E R ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def init_first_layer(self) -> InitialLayer:
        """
        Initialize and return the first layer of the ELM model.

        :return: InitialLayer instance.
        """

        return InitialLayer(train_data=self.train_data,
                            train_labels=self.train_labels,
                            test_data=self.test_data,
                            test_labels=self.test_labels,
                            n_input_nodes=self.gen_ds_cfg.get("num_features"),
                            n_hidden_nodes=self.n_hidden_nodes_layer_1,
                            activation="ReLU",
                            loss="mse",
                            name="phase_1",
                            method=self.cfg.method,
                            bias="zero")

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- T R A I N   F I R S T   L A Y E R ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @measure_execution_time
    def train_first_layer(self) -> None:
        """
        Train the first layer of the ELM model.

        :return: None
        """

        self.initial_layer = self.init_first_layer()
        self.initial_layer.train()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------- E V A L U A T E   F I R S T   L A Y E R ------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def evaluate_first_layer(self) -> None:
        """
        Evaluates the first layer of the ELM model.

        :return: None
        """

        train_metrics = self.initial_layer.evaluate(data=self.train_data,
                                                    labels=self.train_labels,
                                                    metrics=self.metrics,
                                                    operation="train")

        test_metrics = self.initial_layer.evaluate(data=self.test_data,
                                                   labels=self.test_labels,
                                                   metrics=self.metrics,
                                                   operation="test")

        pretty_print_results(metric_results=train_metrics,
                             operation="train",
                             name=self.initial_layer.name,
                             training_time=self.total_execution_time.get("train_first_layer"))

        pretty_print_results(metric_results=test_metrics,
                             name=self.initial_layer.name,
                             operation="test")

        plot_confusion_matrix(metrics=train_metrics,
                              path_to_plot=self.path_to_cm,
                              labels=self.gen_ds_cfg.get("class_labels"),
                              name_of_dataset=self.gen_ds_cfg.get("dataset_name"),
                              operation="train",
                              method=self.cfg.method,
                              phase_name=self.initial_layer.name)

        plot_confusion_matrix(metrics=test_metrics,
                              path_to_plot=self.path_to_cm,
                              labels=self.gen_ds_cfg.get("class_labels"),
                              name_of_dataset=self.gen_ds_cfg.get("dataset_name"),
                              operation="test",
                              method=self.cfg.method,
                              phase_name=self.initial_layer.name)

    def evaluate_additional_layer(self, layer: AdditionalLayer, phase_name: str):
        """

        :param layer:
        :param phase_name:
        :return:
        """

        train_metrics = layer.evaluate(labels=self.train_labels,
                                       metrics=self.metrics,
                                       operation="train")

        test_metrics = layer.evaluate(labels=self.test_labels,
                                      metrics=self.metrics,
                                      operation="test")

        pretty_print_results(metric_results=train_metrics,
                             operation="train",
                             name=layer.name,
                             training_time=self.total_execution_time.get(phase_name))

        pretty_print_results(metric_results=test_metrics,
                             name=layer.name,
                             operation="test")

        plot_confusion_matrix(metrics=train_metrics,
                              path_to_plot=self.path_to_cm,
                              labels=self.gen_ds_cfg.get("class_labels"),
                              name_of_dataset=self.gen_ds_cfg.get("dataset_name"),
                              operation="train",
                              method=self.cfg.method,
                              phase_name=layer.name)

        plot_confusion_matrix(metrics=test_metrics,
                              path_to_plot=self.path_to_cm,
                              labels=self.gen_ds_cfg.get("class_labels"),
                              name_of_dataset=self.gen_ds_cfg.get("dataset_name"),
                              operation="test",
                              method=self.cfg.method,
                              phase_name=layer.name)

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- I N I T   S E C O N D   L A Y E R ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def init_second_layer(self) -> AdditionalLayer:
        """
        Initialize and return the second layer of the ELM model.

        :return: AdditionalLayer instance.
        """

        return AdditionalLayer(previous_layer=self.initial_layer,
                               n_hidden_nodes=self.n_hidden_nodes_layer_2,
                               n_output_nodes=self.n_output_nodes,
                               mu=0,
                               sigma=10,
                               activation="ReLU",
                               loss="mse",
                               name="phase_2",
                               method=self.cfg.method)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------- T R A I N   S E C O N D   L A Y E R --------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @measure_execution_time
    def train_second_layer(self) -> None:
        """
        Train the second layer of the ELM model.

        :return: None
        """

        self.second_layer = self.init_second_layer()
        self.second_layer.train(self.train_labels)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- I N I T   T H I R D   L A Y E R ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def init_third_layer(self) -> AdditionalLayer:
        """
        Initialize and return the third layer of the ELM model.

        :return: AdditionalLayer instance.
        """

        return AdditionalLayer(previous_layer=self.initial_layer,
                               n_hidden_nodes=self.n_hidden_nodes_layer_3,
                               n_output_nodes=self.n_output_nodes,
                               mu=0,
                               sigma=10,
                               activation="ReLU",
                               loss="mse",
                               name="phase_3",
                               method=self.cfg.method)

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- T R A I N   T H I R D   L A Y E R ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @measure_execution_time
    def train_third_layer(self) -> None:
        """
        Train the third layer of the ELM model.

        :return: None
        """

        self.third_layer = self.init_third_layer()
        self.third_layer.train(self.train_labels)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------- G E T   T O T A L   E X E C U T I O N   T I M E --------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_total_execution_time(self) -> float:
        """
        Get the total execution time by summing up the individual phase execution times.

        :return: Total execution time in seconds.
        """

        total_time = sum(self.total_execution_time.values())
        return total_time
