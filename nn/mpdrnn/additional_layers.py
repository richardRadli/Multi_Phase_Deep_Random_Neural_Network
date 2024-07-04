import torch

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, mean_squared_error
from tqdm import tqdm

from nn.models.elm import ELM
from utils.utils import measure_execution_time, pretty_print_results, plot_confusion_matrix, plot_metrics


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++ A D D I T I O N A L   L A Y E R +++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class AdditionalLayer:
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- __I N I T__ ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 previous_layer,
                 train_loader,
                 test_loader,
                 n_hidden_nodes: int,
                 mu: float,
                 sigma: float,
                 activation: str,
                 method: str,
                 phase_name: str,
                 directory_path,
                 general_settings=None,
                 config=None):

        # Initialize attributes
        self.H_prev = previous_layer.save_weights["H_prev"]
        self.previous_weights = previous_layer.save_weights["prev_weights"]
        self.predict_h_train_prev = previous_layer.save_weights["predict_h_train"]
        self.predict_h_test_prev = previous_layer.save_weights["predict_h_test"]
        self.metrics = previous_layer.save_metrics["metrics"]

        self.phase_name = phase_name
        self.method = method
        self.n_hidden_nodes = n_hidden_nodes

        self.H_i_layer = None
        self.H_i_pseudo_inverse = None
        self.predict_h_train = None
        self.predict_h_test = None
        self.new_weights = None

        self.hidden_layer_i = self.create_hidden_layer_i(mu, sigma)

        self.elm = ELM(activation)

        self.train_loader = train_loader
        self.test_loader = test_loader

        if general_settings is not None:
            self.gen_settings = general_settings

        if config is not None:
            self.config = config

        if directory_path is not None:
            self.directory_path = directory_path

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------ C R E A T E   H I D D E N   L A Y E R   I -----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_hidden_layer_i(self, mu, sigma):
        """

        Args:
            mu:
            sigma:

        Returns:

        """

        noise = torch.normal(mean=mu, std=sigma, size=(self.previous_weights.shape[0],
                                                       self.previous_weights.shape[1]))
        w_rnd_out_i = self.previous_weights + noise
        hidden_layer_i_a = torch.hstack((self.previous_weights, w_rnd_out_i))

        if self.method == "BASE":
            w_rnd = torch.normal(mean=mu, std=sigma, size=(self.previous_weights.shape[0],
                                                           self.n_hidden_nodes))
            hidden_layer_i = torch.hstack((hidden_layer_i_a, w_rnd))
        else:
            w_rnd = torch.normal(mean=mu, std=sigma, size=(self.previous_weights.shape[0],
                                                           self.n_hidden_nodes // 2))
            q, _ = torch.linalg.qr(w_rnd)
            orthogonal_matrix = torch.mm(q, q.t())
            hidden_layer_i = torch.cat((hidden_layer_i_a, orthogonal_matrix), dim=1)

        return hidden_layer_i

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- T R A I N ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @measure_execution_time
    def train(self):
        """

        Returns:

        """

        for _, y in tqdm(self.train_loader, total=len(self.train_loader), desc="Training"):
            self.H_i_layer = self.elm(self.H_prev, self.hidden_layer_i)
            self.new_weights = torch.pinverse(self.H_i_layer).matmul(y)

    def collect_metrics(self, operation, accuracy, precision, recall, fscore, loss, cm) -> None:
        """

        Args:
            operation:
            accuracy:
            precision:
            recall:
            fscore:
            loss:
            cm:

        Returns:

        """

        keys = [f'{operation}_accuracy', f'{operation}_precision', f'{operation}_recall',
                f'{operation}_fscore', f'{operation}_loss', f'{operation}_cm']
        values = [accuracy, precision, recall, fscore, loss, cm]

        if operation == "train":
            keys.append(f"{operation}_exe_time")
            values.append(self.train.execution_time)

        for key, value in zip(keys, values):
            if key in self.metrics:
                if not isinstance(self.metrics[key], list):
                    self.metrics[key] = [self.metrics[key]]
                self.metrics[key].append(value)
            else:
                self.metrics[key] = [value]

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- P R E D I C T --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def predict_and_evaluate(self, operation: str):
        """

        Args:
            operation:

        Returns:

        """

        if operation not in ["train", "test"]:
            raise ValueError('An unknown operation \'%s\'.' % operation)

        dataloader = self.train_loader if operation == "train" else self.test_loader

        for x, y in tqdm(dataloader, total=len(dataloader), desc=f"Predicting {operation}"):
            if operation == "train":
                h_n = self.elm(self.predict_h_train_prev, self.hidden_layer_i)
            else:
                h_n = self.elm(self.predict_h_test_prev, self.hidden_layer_i)

            setattr(self, 'predict_h_train' if operation == 'train' else 'predict_h_test', h_n)
            predictions = h_n.matmul(self.new_weights)

            y_predicted_argmax = torch.argmax(predictions, dim=-1)
            y_true_argmax = torch.argmax(y, dim=-1)
            accuracy = accuracy_score(y_true_argmax, y_predicted_argmax)
            precision, recall, fscore, _ = (
                precision_recall_fscore_support(y_true_argmax, y_predicted_argmax, average='macro')
            )
            cm = confusion_matrix(y_true_argmax, y_predicted_argmax)
            loss = mean_squared_error(y_true_argmax, y_predicted_argmax)

            pretty_print_results(
                acc=accuracy,
                precision=precision,
                recall=recall,
                fscore=fscore,
                loss=loss,
                root_dir=self.directory_path.get("results"),
                operation=operation,
                name=self.phase_name,
                exe_time=self.train.execution_time if operation == "train" else None
            )

            self.collect_metrics(operation, accuracy, precision, recall, fscore, loss, cm)

    def plot_results(self, operation: str):
        """

        Args:
            operation:

        Returns:

        """

        plot_confusion_matrix(cm=self.metrics.get(f"{operation}_cm"),
                              path_to_plot=self.directory_path.get("cm"),
                              name_of_dataset=self.config.dataset_name,
                              operation=operation,
                              method=self.config.method,
                              labels=self.gen_settings.get("class_labels"))

        plot_metrics(train=self.metrics.get(f"train_accuracy"),
                     test=self.metrics.get(f"test_accuracy"),
                     metric_type="Accuracy",
                     path_to_plot=self.directory_path.get("metrics"),
                     name_of_dataset=self.config.dataset_name,
                     method=self.config.method, )

        plot_metrics(train=self.metrics.get(f"train_precision"),
                     test=self.metrics.get(f"test_precision"),
                     metric_type="Precision",
                     path_to_plot=self.directory_path.get("metrics"),
                     name_of_dataset=self.config.dataset_name,
                     method=self.config.method,
                     )

        plot_metrics(train=self.metrics.get(f"train_recall"),
                     test=self.metrics.get(f"test_recall"),
                     metric_type="Recall",
                     path_to_plot=self.directory_path.get("metrics"),
                     name_of_dataset=self.config.dataset_name,
                     method=self.config.method,
                     )

        plot_metrics(train=self.metrics.get(f"train_fscore"),
                     test=self.metrics.get(f"test_fscore"),
                     metric_type="F-score",
                     path_to_plot=self.directory_path.get("metrics"),
                     name_of_dataset=self.config.dataset_name,
                     method=self.config.method,
                     )

        plot_metrics(train=self.metrics.get(f"train_loss"),
                     test=self.metrics.get(f"test_loss"),
                     metric_type="Loss",
                     path_to_plot=self.directory_path.get("metrics"),
                     name_of_dataset=self.config.dataset_name,
                     method=self.config.method,
                     )

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- W E I G H T S -------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @property
    def save_weights(self):
        return {
            'prev_weights': self.new_weights,
            'hidden_i_prev': self.hidden_layer_i,
            'H_prev': self.H_i_layer,
            'predict_h_train': self.predict_h_train,
            'predict_h_test': self.predict_h_test
        }

    @property
    def save_metrics(self):
        return {
            "metrics": self.metrics,
        }

    def main(self):
        self.train()
        self.predict_and_evaluate("train")
        self.predict_and_evaluate("test")
        if self.phase_name == "Phase 3":
            self.plot_results("train")
