import torch

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, mean_squared_error
from tqdm import tqdm

from elm import ELM
from utils.utils import measure_execution_time, pretty_print_results


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
                 mu: float = 0,
                 sigma: float = 10,
                 activation: str = "ReLU",
                 method: str = "BASE"):

        # Initialize attributes
        self.H_prev = previous_layer.weights["H_prev"]
        self.previous_weights = previous_layer.weights["prev_weights"]
        self.predict_h_train_prev = previous_layer.weights["predict_h_train"]
        self.predict_h_test_prev = previous_layer.weights["predict_h_test"]

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

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------ C R E A T E   H I D D E N   L A Y E R   I -----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_hidden_layer_i(self, mu, sigma):
        noise = torch.normal(mean=mu, std=sigma, size=(self.previous_weights.shape[0], self.previous_weights.shape[1]))
        w_rnd_out_i = self.previous_weights + noise
        hidden_layer_i_a = torch.hstack((self.previous_weights, w_rnd_out_i))

        w_rnd = torch.normal(mean=mu, std=sigma, size=(self.previous_weights.shape[0], self.n_hidden_nodes))
        hidden_layer_i = torch.hstack((hidden_layer_i_a, w_rnd))

        return hidden_layer_i

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- C A L C   O R T ------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def calc_ort(x):
        x = x.reshape(x.shape[0], 1)
        q, _ = torch.qr(torch.tensor(x))
        return q

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- T R A I N ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @measure_execution_time
    def train(self):
        for _, y in tqdm(self.train_loader, total=len(self.train_loader), desc="Training"):
            self.H_i_layer = self.elm(self.H_prev, self.hidden_layer_i)
            self.new_weights = torch.pinverse(self.H_i_layer).matmul(y)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- P R E D I C T --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def predict_and_evaluate(self, operation: str):
        if operation not in ["train", "test"]:
            raise ValueError('An unknown operation \'%s\'.' % operation)

        if operation == "train":
            dataloader = self.train_loader
        else:
            dataloader = self.test_loader

        for x, y in tqdm(dataloader, total=len(dataloader), desc=f"Predicting {operation}"):
            if operation == "train":
                H_n = self.elm(self.predict_h_train_prev, self.hidden_layer_i)
            else:
                H_n = self.elm(self.predict_h_test_prev, self.hidden_layer_i)

            setattr(self, 'predict_h_train' if operation == 'train' else 'predict_h_test', H_n)
            predictions = H_n.matmul(self.new_weights)

            y_predicted_argmax = torch.argmax(predictions, dim=-1)
            y_true_argmax = torch.argmax(y, dim=-1)
            accuracy = accuracy_score(y_true_argmax, y_predicted_argmax)
            precision, recall, fscore, _ = (
                precision_recall_fscore_support(y_true_argmax, y_predicted_argmax, average='macro')
            )
            cm = confusion_matrix(y_true_argmax, y_predicted_argmax)
            loss = mean_squared_error(y_true_argmax, y_predicted_argmax)

            pretty_print_results(accuracy, precision, recall, fscore, loss, operation, "phase_2")

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- W E I G H T S -------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @property
    def weights(self):
        return {
            'prev_weights': self.new_weights,
            'hidden_i_prev': self.hidden_layer_i,
            'H_prev': self.H_i_layer,
            'predict_h_train': self.predict_h_train,
            'predict_h_test': self.predict_h_test
        }

    def main(self):
        self.train()
        self.predict_and_evaluate("train")
        self.predict_and_evaluate("test")
