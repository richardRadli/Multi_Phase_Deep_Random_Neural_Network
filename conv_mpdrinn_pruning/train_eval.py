import logging
import numpy as np
import os
import torch

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST, FashionMNIST

from conv_mpdrinn_pruning.conv_slfn import ConvolutionalSLFN, SecondLayer, ThirdLayer
from config.json_config import json_config_selector
from config.dataset_config import drnn_paths_config, general_dataset_configs
from utils.utils import log_to_excel, load_config_json, setup_logger, create_timestamp


class TrainEval:
    def __init__(self) -> None:
        setup_logger()
        self.cfg = (
            load_config_json(
                json_schema_filename=json_config_selector("cipmpdrnn").get("schema"),
                json_filename=json_config_selector("cipmpdrnn").get("config")
            )
        )
        timestamp = create_timestamp()
        self.save_path = (
            os.path.join(
                drnn_paths_config(self.cfg["dataset_name"])["cipmpdrnn"]["path_to_results"],
                f"{self.cfg['dataset_name']}_{timestamp}.xlsx"
            )
        )

        train_dataset, test_dataset = self.load_dataset()
        sample_image = next(iter(train_dataset))

        self.train_loader = (
            DataLoader(
                dataset=train_dataset,
                batch_size=len(train_dataset),
                shuffle=False
            )
        )
        self.test_loader = (
            DataLoader(
                dataset=test_dataset,
                batch_size=len(test_dataset),
                shuffle=False
            )
        )

        self.dataset_info = {
            "width": sample_image[0].size(1),
            "height": sample_image[0].size(2),
            "in_channels": sample_image[0].size(0),
            "num_train_images": len(train_dataset),
            "out_channels": len(np.unique(train_dataset.targets.data))
        }

        self.settings = {
            "num_filters": self.cfg["num_filters"],
            "conv_kernel_size": self.cfg["conv_kernel_size"],
            "stride": self.cfg["stride"],
            "padding": self.cfg["padding"],
            "pool_kernel_size": self.cfg["pool_kernel_size"],
            "pruning_percentage": self.cfg["pruning_percentage"],
            "rcond": self.cfg["rcond"],
            "mu": self.cfg["mu"],
            "sigma": self.cfg["sigma"],
            "penalty_term": self.cfg["penalty"],
            "hidden_nodes": self.cfg["hidden_nodes"]
        }

    def load_dataset(self, dataset_type='mnist'):
        if dataset_type.lower() == 'mnist':
            dataset_class = MNIST
        elif dataset_type.lower() == 'fashion_mnist':
            dataset_class = FashionMNIST
        else:
            raise ValueError("Invalid dataset type. Choose either 'mnist' or 'fashion_mnist'.")

        path_to_dataset = general_dataset_configs(dataset_type=self.cfg["dataset_name"])["original_dataset"]
        train_dataset = dataset_class(
            root=path_to_dataset,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        )

        test_dataset = dataset_class(
            root=path_to_dataset,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        )

        return train_dataset, test_dataset

    def train_eval_main_net(self):
        cnn_slfn = (
            ConvolutionalSLFN(
                self.settings,
                self.dataset_info
            )
        )

        cnn_slfn.train_net(self.train_loader)
        cnn_slfn.test_net(cnn_slfn, self.test_loader, cnn_slfn.beta_weights, 1)

        _, least_important_filters = (
            cnn_slfn.pruning_first_layer()
        )

        return cnn_slfn, least_important_filters

    def train_auxiliary_net(self):
        aux_cnn_slfn = (
            ConvolutionalSLFN(
                self.settings,
                self.dataset_info
            )
        )
        aux_cnn_slfn.train_net(self.train_loader)
        most_important_filters, _ = (
            aux_cnn_slfn.pruning_first_layer()
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
                cnn_slfn
            )
        )
        second_layer.train_fc_layer(self.train_loader)
        second_layer.test_net(
            base_instance=cnn_slfn,
            data_loader=self.test_loader,
            layer_weights=[second_layer.extended_beta_weights,
                           second_layer.gamma_weights],
            num_layers=2
        )

        _, least_important_indices = (
            second_layer.pruning()
        )

        return second_layer, least_important_indices

    def train_prune_second_layer_auxiliary(self, cnn_slfn):
        second_layer_aux = (
            SecondLayer(
                base_instance=cnn_slfn
            )
        )
        second_layer_aux.train_fc_layer(self.train_loader)
        most_important_indices, _ = (
            second_layer_aux.pruning()
        )

        return second_layer_aux, most_important_indices

    def replace_retrain_second_layer(
            self, cnn_slfn, main_layer, aux_layer, most_important_indices, least_important_indices, weight_attr
    ):
        best_weight_tensor = getattr(aux_layer, weight_attr).data
        best_weights = best_weight_tensor[:, most_important_indices]
        weight_tensor = getattr(main_layer, weight_attr).data
        weight_tensor[:, least_important_indices] = best_weights

        main_layer.train_fc_layer(self.train_loader)
        main_layer.test_net(
            base_instance=cnn_slfn,
            data_loader=self.test_loader,
            layer_weights=[main_layer.extended_beta_weights,
                           main_layer.gamma_weights],
            num_layers=2
        )

        return main_layer

    def train_eval_third_layer(self, cnn_slfn, second_layer):
        third_layer = (
            ThirdLayer(
                second_layer
            )
        )
        third_layer.train_fc_layer(self.train_loader)
        third_layer.test_net(
            base_instance=cnn_slfn,
            data_loader=self.test_loader,
            layer_weights=[third_layer.extended_beta_weights,
                           third_layer.extended_gamma_weights,
                           third_layer.delta_weights],
            num_layers=3
        )

        _, least_important_indices = third_layer.pruning()

        return third_layer, least_important_indices

    def train_prune_third_layer_auxiliary(self, second_layer):
        third_layer_aux = (
            ThirdLayer(
                second_layer
            )
        )
        third_layer_aux.train_fc_layer(self.train_loader)
        most_important_indices, _ = third_layer_aux.pruning()

        return third_layer_aux, most_important_indices

    def replace_retrain_third_layer(
            self, cnn_slfn, main_layer, aux_layer, most_important_indices, least_important_indices, weight_attr
    ):
        best_weight_tensor = getattr(aux_layer, weight_attr).data
        best_weights = best_weight_tensor[:, most_important_indices]
        weight_tensor = getattr(main_layer, weight_attr).data
        weight_tensor[:, least_important_indices] = best_weights

        main_layer.train_fc_layer(self.train_loader)
        main_layer.test_net(cnn_slfn,
                            self.test_loader,
                            [main_layer.extended_beta_weights,
                             main_layer.extended_gamma_weights,
                             main_layer.delta_weights],
                            3)

        return main_layer

    def main(self):
        if self.cfg["seed"]:
            torch.manual_seed(seed=42)

        for i in range(self.cfg["number_of_tests"]):
            logging.info(f"\nActual iteration: {i+1}\n")
            execution_time = []

            # First layer
            cnn_slfn, least_important_filters = (
                self.train_eval_main_net()
            )
            execution_time.append(cnn_slfn.train_net.execution_time)
            aux_cnn_slfn, most_important_filters = (
                self.train_auxiliary_net()
            )
            execution_time.append(aux_cnn_slfn.train_net.execution_time)
            cnn_slfn = (
                self.replace_retrain(
                    cnn_slfn, aux_cnn_slfn, least_important_filters, most_important_filters
                )
            )
            execution_time.append(cnn_slfn.train_net.execution_time)

            # Second layer
            second_layer, least_important_indices = (
                self.train_eval_second_layer(
                    cnn_slfn
                )
            )
            execution_time.append(second_layer.train_fc_layer.execution_time)
            second_layer_aux, most_important_indices = (
                self.train_prune_second_layer_auxiliary(
                    cnn_slfn
                )
            )
            execution_time.append(second_layer_aux.train_fc_layer.execution_time)
            second_layer = (
                self.replace_retrain_second_layer(
                    cnn_slfn,
                    second_layer,
                    second_layer_aux,
                    most_important_indices,
                    least_important_indices,
                    "extended_beta_weights"
                )
            )
            execution_time.append(second_layer.train_fc_layer.execution_time)

            # Third layer
            third_layer, least_important_indices = (
                self.train_eval_third_layer(
                    cnn_slfn,
                    second_layer
                )
            )
            execution_time.append(third_layer.train_fc_layer.execution_time)
            third_layer_aux, most_important_indices = (
                self.train_prune_third_layer_auxiliary(
                    second_layer
                )
            )
            execution_time.append(third_layer_aux.train_fc_layer.execution_time)
            third_layer = self.replace_retrain_third_layer(
                cnn_slfn, third_layer, third_layer_aux, most_important_indices, least_important_indices,
                "extended_gamma_weights"
            )
            execution_time.append(third_layer.train_fc_layer.execution_time)

            logging.info(f"Total execution time: {sum(execution_time)}")
            logging.info(f"Final accuracy: {third_layer.accuracy[-1]}")

            log_to_excel(execution_time, third_layer.accuracy[-1], self.save_path)

            execution_time.clear()


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    num_filters=16, kernel_size=7, stride=1, padding=1, pool_kernel_size=2, pruning_percentage=0.2
    96.86% -> 97.08%
    97.08% -> 97.51%
    97.51% -> 97.60%
    97.60% -> 97.68%
    97.68% -> 97.71%
    
    neurons
    2304 - 1500 - 500
    """

    try:
        train_eval = TrainEval()
        train_eval.main()
    except KeyboardInterrupt:
        logging.error("Keyboard Interrupt")
