import numpy as np
import time

from scipy import linalg, stats
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

from elm.src.config.config import DatasetConfig, MPDRNNConfig
from elm.src.config.dataset_config import general_dataset_configs


class HELM:
    def __init__(self, penalty, scaling_factor):
        cfg = MPDRNNConfig().parse()
        cfg_data_preprocessing = DatasetConfig().parse()
        gen_ds_cfg = general_dataset_configs(cfg)

        data_file = gen_ds_cfg.get("cached_dataset_file")
        data = np.load(data_file, allow_pickle=True)

        self.train_data = data[0]
        self.train_labels = data[2]
        self.test_data = data[1]
        self.test_labels = data[3]

        if cfg_data_preprocessing.normalize:
            if cfg_data_preprocessing.type_of_normalization == "zscore":
                self.train_data = stats.zscore(self.train_data)
                self.test_data = stats.zscore(self.test_data)
            elif cfg_data_preprocessing.type_of_normalization == "minmax":
                scaler = preprocessing.MinMaxScaler()
                self.train_data = scaler.fit_transform(self.train_data)
                self.test_data = scaler.fit_transform(self.test_data)
            else:
                raise ValueError("Wrong type of normalization!")

        if cfg.seed:
            np.random.seed(1234)

        self.random_weights_1 = 2 * np.random.rand(self.train_data.shape[1] + 1,
                                                   gen_ds_cfg.get("helm_neurons")[0]) - 1
        self.random_weights_2 = 2 * np.random.rand(gen_ds_cfg.get("helm_neurons")[0] + 1,
                                                   gen_ds_cfg.get("helm_neurons")[1]) - 1
        self.random_weights_3 = (2 * np.random.rand(gen_ds_cfg.get("helm_neurons")[1] + 1,
                                                    gen_ds_cfg.get("helm_neurons")[2]) - 1).T
        self.random_weights_3 = linalg.orth(self.random_weights_3).T

        self.C_penalty = penalty
        self.scaling_factor = scaling_factor

    @staticmethod
    def sparse_elm_autoencoder(a, b, lam, itrs):
        """

        param a: a matrix with size (d, n), where d is the dimension of the input data and n is the number of training
                 samples.
        param b: a matrix with size (d, m), where m is the number of hidden neurons in the autoencoder.
        param lam: a scalar that controls the sparsity of the learned representation.
        param itrs: the number of iterations for training the autoencoder.
        :return: The function returns a matrix "x" with size (m, n), which is the learned representation of the input
                 data.
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
    def min_max_scale(matrix, scale: str):
        """
        The function scales the values of the matrix to either the range of [-1, 1] or [0, 1] based on the value of
        the parameter "scale". The function returns the scaled data along with a list that contains the minimum values,
        maximum values, and ranges for each row.

        param matrix:
        param scale:
        :return:
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
        The purpose of this function is to normalize the input data based on the given minimum values and range values.

        param data:
        param min_values:
        param range_values:
        :return:
        """

        return (data - min_values) / range_values

    def train(self):
        """

        :return:
        """

        start = time.time()
        # First layer RELM
        h1 = np.hstack([self.train_data, np.ones((self.train_data.shape[0], 1)) * 0.1])
        a1 = h1 @ self.random_weights_1
        a1, _ = self.min_max_scale(a1, "-1_1")
        beta1 = self.sparse_elm_autoencoder(a=a1, b=h1, lam=1e-3, itrs=50)
        del a1
        t1 = h1 @ beta1.T
        del h1
        print("Layer 1: Max Val of Output", np.max(t1), "Min Val", np.min(t1))
        t1, ps1 = self.min_max_scale(t1.T, "0_1")

        # Second layer RELM
        h2 = np.hstack([t1.T, np.ones((t1.T.shape[0], 1)) * 0.1])
        del t1
        a2 = h2 @ self.random_weights_2
        a2, _ = self.min_max_scale(a2, "-1_1")
        beta2 = self.sparse_elm_autoencoder(a=a2, b=h2, lam=1e-3, itrs=50)
        del a2
        t2 = h2 @ beta2.T
        del h2
        print("Layer 2: Max Val of Output", np.max(t2), "Min Val", np.min(t2))
        t2, ps2 = self.min_max_scale(t2.T, "0_1")

        # Original ELM
        h3 = np.hstack([t2.T, np.ones((t2.T.shape[0], 1)) * 0.1])
        del t2
        t3 = h3 @ self.random_weights_3
        del h3
        l3 = np.amax(np.amax(t3))
        l3 = self.scaling_factor / l3
        print("Layer 3: Max Val of Output %f Min Val %f" % (l3, np.amin(t3)))

        t3 = np.tanh(t3 * l3)

        # Finsh Training
        beta = np.linalg.solve(t3.T.dot(t3) + np.eye(t3.shape[1]) * self.C_penalty, t3.T.dot(self.train_labels))
        end = time.time()
        print('Training has been finished!')
        print(f'The Total Training Time is : {end - start} seconds')

        return t3, beta, beta1, beta2, l3, ps1, ps2

    def training_accuracy(self, t3, beta):
        """

        param t3:
        param beta:
        :return:
        """

        y_predicted = t3 @ beta
        y_predicted_argmax = np.argmax(np.asarray(y_predicted), axis=-1)
        y_true_argmax = np.argmax(np.asarray(self.train_labels), axis=-1)
        training_accuracy = accuracy_score(y_true_argmax, y_predicted_argmax)
        print(f"Training Accuracy is: {training_accuracy * 100}%")

    def testing_accuracy(self, beta, beta1, beta2, l3, ps1, ps2):
        """

        param beta:
        param beta1:
        param beta2:
        param l3:
        param ps1:
        param ps2:
        :return:
        """

        #  First layer feedforward
        start = time.time()
        hh1 = np.hstack([self.test_data, np.ones((self.test_data.shape[0], 1)) * 0.1])
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
        y_true_argmax = np.argmax(np.asarray(self.test_labels), axis=-1)
        testing_accuracy = accuracy_score(y_true_argmax, y_predicted_argmax)
        end = time.time()

        print("Testing has been finished!")
        print(f"The Total Testing Time is:  {end - start} seconds")
        print(f"Testing Accuracy is: {testing_accuracy * 100}%")

    def main(self):
        t3, beta, beta1, beta2, l3, ps1, ps2 = self.train()
        self.training_accuracy(t3, beta)
        self.testing_accuracy(beta, beta1, beta2, l3, ps1, ps2)


if __name__ == "__main__":
    helm = HELM(penalty=2 ** -30, scaling_factor=0.8)
    helm.main()
