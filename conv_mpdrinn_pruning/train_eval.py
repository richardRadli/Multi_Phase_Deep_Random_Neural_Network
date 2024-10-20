import logging
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST

from conv_mpdrinn_pruning.conv_slfn import ConvolutionalSLFN, MultiPhaseDeepRandomizedNeuralNetworkSubsequent
from utils.utils import setup_logger


class TrainEval:
    def __init__(self, num_filters, kernel_size, stride, padding, pool_kernel_size, pruning_percentage, rcond) -> None:
        setup_logger()
        train_dataset, test_dataset = self.load_dataset()
        sample_image = next(iter(train_dataset))

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

        self.dataset_info = {
            "width": sample_image[0].size(1),
            "height": sample_image[0].size(2),
            "in_channels": sample_image[0].size(0),
            "num_train_images": len(train_dataset),
            "out_channels": len(np.unique(train_dataset.targets.data))
        }

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_kernel_size = pool_kernel_size
        self.pruning_percentage = pruning_percentage
        self.rcond = rcond

    @staticmethod
    def load_dataset():
        train_dataset = MNIST(
            root="C:/Users/ricsi/Desktop/",
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor()])
        )

        test_dataset = MNIST(
            root="C:/Users/ricsi/Desktop/",
            train=False,
            download=False,
            transform=transforms.Compose([transforms.ToTensor()])
        )

        return train_dataset, test_dataset

    @staticmethod
    def seed_everything(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train_eval_main_net(self):
        cnn_slfn = (
            ConvolutionalSLFN(
                self.num_filters,
                self.kernel_size,
                self.stride,
                self.padding,
                self.pool_kernel_size,
                self.rcond,
                self.dataset_info
            )
        )

        cnn_slfn.train_net(self.train_loader)
        cnn_slfn.test_net(cnn_slfn, self.test_loader, cnn_slfn.beta_weights, 1)

        _, least_important_filters = (
            cnn_slfn.pruning(
                self.pruning_percentage
            )
        )

        return cnn_slfn, least_important_filters

    def train_auxiliary_net(self):
        aux_cnn_slfn = (
            ConvolutionalSLFN(
                self.num_filters,
                self.kernel_size,
                self.stride,
                self.padding,
                self.pool_kernel_size,
                self.rcond,
                self.dataset_info
            )
        )
        aux_cnn_slfn.train_net(self.train_loader)
        most_important_filters, _ = (
            aux_cnn_slfn.pruning(
                self.pruning_percentage
            )
        )

        return aux_cnn_slfn, most_important_filters

    def replace_retrain(self, cnn_slfn, aux_cnn_slfn, least_important_filters, most_important_filters):
        cnn_slfn = cnn_slfn.replace_filters(cnn_slfn, aux_cnn_slfn, least_important_filters, most_important_filters)
        cnn_slfn.train_net(self.train_loader)
        cnn_slfn.test_net(cnn_slfn, self.test_loader, cnn_slfn.beta_weights, 1)

        return cnn_slfn

    def create_second_layer(self, cnn_slfn):
        second_layer = (
            MultiPhaseDeepRandomizedNeuralNetworkSubsequent(
                cnn_slfn,
                mu=0.0,
                sigma=0.1,
                penalty_term=0.15,
                n_hidden_nodes=1500
            )
        )
        second_layer.train_second_layer(self.train_loader)
        second_layer.test_net(cnn_slfn,
                              self.test_loader,
                              [second_layer.extended_beta_weights,
                               second_layer.gamma_weights],
                              2)

    def main(self):
        self.seed_everything()

        cnn_slfn, least_important_filters = self.train_eval_main_net()
        aux_cnn_slfn, most_important_filters = self.train_auxiliary_net()
        cnn_slfn = self.replace_retrain(cnn_slfn, aux_cnn_slfn, least_important_filters, most_important_filters)
        self.create_second_layer(cnn_slfn)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    num_filters=16, kernel_size=7, stride=1, padding=1, pool_kernel_size=2, pruning_percentage=0.2
    96.86% -> 97.13%
    """

    train_eval = (
        TrainEval(
            num_filters=16,
            kernel_size=7,
            stride=1,
            padding=1,
            pool_kernel_size=2,
            pruning_percentage=0.2,
            rcond=1e-15
        )
    )
    train_eval.main()
