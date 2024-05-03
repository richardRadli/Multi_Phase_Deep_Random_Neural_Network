import torch

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, mean_squared_error
from tqdm import tqdm

from config.config import MPDRNNConfig
from elm import ELM
from utils.utils import measure_execution_time, pretty_print_results


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++ I N I T I A L   L A Y E R ++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class InitialLayer:
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- __I N I T__ ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 n_input_nodes: int,
                 n_hidden_nodes: int,
                 train_loader,
                 test_loader,
                 activation: str,
                 method: str,
                 phase_name: str,
                 directory_path):
        self.cfg = MPDRNNConfig().parse()
        self.method = method
        self.phase_name = phase_name

        self.train_loader = train_loader
        self.test_loader = test_loader

        # Variables for the first phase
        self.H1 = None
        self.H_pseudo_inverse = None
        self.beta_weights = None

        # Variables to store prediction results
        self.predict_h_train = None
        self.predict_h_test = None

        self.elm = ELM(activation_function=activation)
        self.alpha_weights = torch.nn.Parameter(torch.randn(n_input_nodes, n_hidden_nodes))

        self.metrics = {}
        self.directory_path = directory_path

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- T R A I N ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @measure_execution_time
    def train(self) -> None:
        """

        Returns:

        """

        for train_x, train_y in tqdm(self.train_loader, total=len(self.train_loader), desc="Training"):
            train_x = train_x
            train_y = train_y
            self.H1 = self.elm(train_x, self.alpha_weights)
            if self.method in ["BASE", "EXP_ORT"]:
                self.beta_weights = torch.pinverse(self.H1).matmul(train_y)
            else:
                identity_matrix = torch.eye(self.H1.shape[1])
                if self.H1.shape[0] > self.H1.shape[1]:
                    self.beta_weights = torch.pinverse(
                        self.H1.T.matmul(self.H1) + identity_matrix / self.cfg.C
                    ).matmul(self.H1.T.matmul(train_y))
                elif self.H1.shape[0] < self.H1.shape[1]:
                    self.beta_weights = self.H1.T.matmul(
                        torch.pinverse(
                            self.H1.matmul(self.H1.T) + identity_matrix / self.cfg.C
                        ).matmul(train_y)
                    )

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

        self.metrics[f'{operation}_accuracy'] = accuracy
        self.metrics[f'{operation}_precision'] = precision
        self.metrics[f'{operation}_recall'] = recall
        self.metrics[f'{operation}_fscore'] = fscore
        self.metrics[f'{operation}_loss'] = loss
        self.metrics[f'{operation}_cm'] = cm

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

        if operation == "train":
            dataloader = self.train_loader
        else:
            dataloader = self.test_loader

        for x, y in tqdm(dataloader, total=len(dataloader), desc=f"Predicting {operation}"):
            x = x
            y = y
            h1 = self.elm(x, self.alpha_weights)
            setattr(self, 'predict_h_train' if operation == 'train' else 'predict_h_test', h1)
            predictions = h1.matmul(self.beta_weights)

            y_predicted_argmax = torch.argmax(predictions, dim=-1)
            y_true_argmax = torch.argmax(y, dim=-1)
            accuracy = accuracy_score(y_true_argmax, y_predicted_argmax)
            precision, recall, fscore, _ = (
                precision_recall_fscore_support(y_true_argmax, y_predicted_argmax, average='macro')
            )
            cm = confusion_matrix(y_true_argmax, y_predicted_argmax)
            loss = mean_squared_error(y_true_argmax, y_predicted_argmax)

            pretty_print_results(
                acc=accuracy, precision=precision, recall=recall, fscore=fscore, loss=loss,
                root_dir=self.directory_path.get("results"),
                operation=operation, name=self.phase_name
            )

            self.collect_metrics(operation, accuracy, precision, recall, fscore, loss, cm)

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- W E I G H T S -------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @property
    def save_weights(self):
        """
        Function to save weights.

        Returns:
             Values of the weights.
        """

        return {
            'prev_weights': self.beta_weights,
            'H_prev': self.H1,
            'predict_h_train': self.predict_h_train,
            'predict_h_test': self.predict_h_test,
        }

    @property
    def save_metrics(self):
        """

        Returns:

        """

        return {
            "metrics": self.metrics,
        }

    def main(self):
        """

        Returns:

        """

        self.train()
        self.predict_and_evaluate(operation="train")
        self.predict_and_evaluate(operation="test")
