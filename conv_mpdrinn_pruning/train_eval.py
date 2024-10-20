import logging
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST

from conv_mpdrinn_pruning.conv_slfn import ConvolutionalSLFN, SecondLayer, ThirdLayer
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
            cnn_slfn.pruning_first_layer(
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
            aux_cnn_slfn.pruning_first_layer(
                self.pruning_percentage
            )
        )

        return aux_cnn_slfn, most_important_filters

    def replace_retrain(self, cnn_slfn, aux_cnn_slfn, least_important_filters, most_important_filters):
        cnn_slfn = cnn_slfn.replace_filters(cnn_slfn, aux_cnn_slfn, least_important_filters, most_important_filters)
        cnn_slfn.train_net(self.train_loader)
        cnn_slfn.test_net(cnn_slfn, self.test_loader, cnn_slfn.beta_weights, 1)

        return cnn_slfn

    def train_eval_second_layer(self, cnn_slfn):
        second_layer = (
            SecondLayer(
                cnn_slfn,
                mu=0.0,
                sigma=0.1,
                penalty_term=0.15,
                n_hidden_nodes=1500
            )
        )
        second_layer.train_fcnn_layer(self.train_loader)
        second_layer.test_net(cnn_slfn,
                              self.test_loader,
                              [second_layer.extended_beta_weights,
                               second_layer.gamma_weights],
                              2)

        _, least_important_indices = second_layer.pruning(pruning_percentage=0.2)

        return second_layer, least_important_indices

    def train_prune_second_layer_auxiliary(self, cnn_slfn):
        second_layer_aux = (
            SecondLayer(
                cnn_slfn,
                mu=0.0,
                sigma=0.1,
                penalty_term=0.15,
                n_hidden_nodes=1500
            )
        )
        second_layer_aux.train_fcnn_layer(self.train_loader)
        most_important_indices, _ = second_layer_aux.pruning(pruning_percentage=0.2)

        return second_layer_aux, most_important_indices

    def replace_retrain_second_layer(
            self, cnn_slfn, main_layer, aux_layer, most_important_indices, least_important_indices, weight_attr
    ):
        best_weight_tensor = getattr(aux_layer, weight_attr).data
        best_weights = best_weight_tensor[:, most_important_indices]
        weight_tensor = getattr(main_layer, weight_attr).data
        weight_tensor[:, least_important_indices] = best_weights

        main_layer.train_fcnn_layer(self.train_loader)
        main_layer.test_net(cnn_slfn,
                            self.test_loader,
                            [main_layer.extended_beta_weights,
                             main_layer.gamma_weights],
                            2)

        return main_layer

    def train_eval_third_layer(self, cnn_slfn, second_layer):
        third_layer = (
            ThirdLayer(
                second_layer,
                500
            )
        )
        third_layer.train_fcnn_layer(self.train_loader)
        third_layer.test_net(cnn_slfn,
                             self.test_loader,
                             [third_layer.extended_beta_weights,
                              third_layer.extended_gamma_weights,
                              third_layer.delta_weights],
                             3)

        _, least_important_indices = third_layer.pruning(pruning_percentage=0.2)

        return third_layer, least_important_indices

    def train_prune_third_layer_auxiliary(self, second_layer):
        third_layer_aux = (
            ThirdLayer(
                second_layer,
                500
            )
        )
        third_layer_aux.train_fcnn_layer(self.train_loader)
        most_important_indices, _ = third_layer_aux.pruning(pruning_percentage=0.2)

        return third_layer_aux, most_important_indices

    def replace_retrain_third_layer(
            self, cnn_slfn, main_layer, aux_layer, most_important_indices, least_important_indices, weight_attr
    ):
        best_weight_tensor = getattr(aux_layer, weight_attr).data
        best_weights = best_weight_tensor[:, most_important_indices]
        weight_tensor = getattr(main_layer, weight_attr).data
        weight_tensor[:, least_important_indices] = best_weights

        main_layer.train_fcnn_layer(self.train_loader)
        main_layer.test_net(cnn_slfn,
                            self.test_loader,
                            [main_layer.extended_beta_weights,
                             main_layer.extended_gamma_weights,
                             main_layer.delta_weights],
                            3)

        return main_layer

    def main(self):
        self.seed_everything()

        cnn_slfn, least_important_filters = self.train_eval_main_net()
        aux_cnn_slfn, most_important_filters = self.train_auxiliary_net()
        cnn_slfn = self.replace_retrain(cnn_slfn, aux_cnn_slfn, least_important_filters, most_important_filters)

        second_layer, least_important_indices = self.train_eval_second_layer(cnn_slfn)
        second_layer_aux, most_important_indices = self.train_prune_second_layer_auxiliary(cnn_slfn)
        second_layer = self.replace_retrain_second_layer(
            cnn_slfn, second_layer, second_layer_aux, most_important_indices, least_important_indices,
            "extended_beta_weights"
        )

        third_layer, least_important_indices = self.train_eval_third_layer(cnn_slfn, second_layer)
        third_layer_aux, most_important_indices = self.train_prune_third_layer_auxiliary(second_layer)
        _ = self.replace_retrain_third_layer(
            cnn_slfn, third_layer, third_layer_aux, most_important_indices, least_important_indices,
            "extended_gamma_weights"
        )


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    num_filters=16, kernel_size=7, stride=1, padding=1, pool_kernel_size=2, pruning_percentage=0.2
    96.86% -> 97.08%
    97.08% -> 97.51%
    97.51% -> 97.60%
    """

    try:
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
    except KeyboardInterrupt as kie:
        logging.error(kie)
