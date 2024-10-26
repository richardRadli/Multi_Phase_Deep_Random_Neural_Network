import logging
import multiprocessing as mp
import os
import ray

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, SVHN

from conv_mpdrinn_pruning.conv_slfn import ConvolutionalSLFN, SecondLayer, ThirdLayer
from config.json_config import json_config_selector
from config.dataset_config import drnn_paths_config, general_dataset_configs
from utils.utils import load_config_json, setup_logger, create_timestamp


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

        train_dataset, test_dataset = self.load_dataset(self.cfg["dataset_name"])
        sample_image = next(iter(train_dataset))

        train_loader = (
            DataLoader(
                dataset=train_dataset,
                batch_size=len(train_dataset),
                shuffle=False
            )
        )
        test_loader = (
            DataLoader(
                dataset=test_dataset,
                batch_size=len(test_dataset),
                shuffle=False
            )
        )

        train_loader = ray.put(train_loader)
        test_loader = ray.put(test_loader)

        self.dataset_info = {
            "width": sample_image[0].size(1),
            "height": sample_image[0].size(2),
            "in_channels": sample_image[0].size(0),
            "num_train_images": len(train_dataset),
            "out_channels": 10  # len(np.unique(train_dataset.targets.data))
        }

        self.settings = {
            # Fixed settings
            "stride": self.cfg["stride"],
            "padding": self.cfg["padding"],
            "pool_kernel_size": self.cfg["pool_kernel_size"],
            "mu": self.cfg["mu"],
            "sigma": self.cfg["sigma"],
            "hidden_nodes": self.cfg["hidden_nodes"],

            "train_loader": train_loader,
            "test_loader": test_loader,

            # Hyperparameters to tune
            "num_filters": tune.grid_search([12, 14, 16, 18, 20]),
            "conv_kernel_size": tune.grid_search([3, 5, 7]),
            "pruning_percentage": tune.uniform(0.1, 1.0),
            "rcond": tune.loguniform(1e-30, 1e-1),
            "penalty_term": tune.uniform(0.1, 45)
        }

    @staticmethod
    def select_dataset(dataset_type):
        dataset_dir = {
            "mnist": MNIST,
            "mnist_fashion": FashionMNIST,
            "cifar10": CIFAR10,
            "svhn": SVHN
        }
        return dataset_dir[dataset_type]

    def load_dataset(self, dataset_type):
        dataset_class = self.select_dataset(dataset_type)

        path_to_dataset = general_dataset_configs(dataset_type=self.cfg["dataset_name"])["original_dataset"]
        train_dataset = (
            dataset_class(
                root=path_to_dataset,
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()])
            )
        )

        test_dataset = (
            dataset_class(
                root=path_to_dataset,
                train=False,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()])
            )
        )

        return train_dataset, test_dataset

    def train_eval_main_net(self, config, train_loader, test_loader):
        cnn_slfn = (
            ConvolutionalSLFN(
                config,
                self.dataset_info
            )
        )

        cnn_slfn.train_net(train_loader)
        cnn_slfn.test_net(cnn_slfn, test_loader, cnn_slfn.beta_weights, 1)

        _, least_important_filters = (
            cnn_slfn.pruning_first_layer()
        )

        return cnn_slfn, least_important_filters

    def train_auxiliary_net(self, config, train_loader):
        aux_cnn_slfn = (
            ConvolutionalSLFN(
                config,
                self.dataset_info
            )
        )
        aux_cnn_slfn.train_net(train_loader)
        most_important_filters, _ = (
            aux_cnn_slfn.pruning_first_layer()
        )

        return aux_cnn_slfn, most_important_filters

    @staticmethod
    def replace_retrain(
            cnn_slfn, aux_cnn_slfn, least_important_filters, most_important_filters, train_loader, test_loader
    ):
        cnn_slfn = cnn_slfn.replace_filters(cnn_slfn, aux_cnn_slfn, least_important_filters, most_important_filters)
        cnn_slfn.train_net(train_loader)
        cnn_slfn.test_net(cnn_slfn, test_loader, cnn_slfn.beta_weights, 1)

        return cnn_slfn

    @staticmethod
    def train_eval_second_layer(cnn_slfn, train_loader, test_loader):
        second_layer = (
            SecondLayer(
                cnn_slfn
            )
        )
        second_layer.train_fc_layer(train_loader)
        second_layer.test_net(
            base_instance=cnn_slfn,
            data_loader=test_loader,
            layer_weights=[second_layer.extended_beta_weights,
                           second_layer.gamma_weights],
            num_layers=2
        )

        _, least_important_indices = (
            second_layer.pruning()
        )

        return second_layer, least_important_indices

    @staticmethod
    def train_prune_second_layer_auxiliary(cnn_slfn, train_loader):
        second_layer_aux = (
            SecondLayer(
                base_instance=cnn_slfn
            )
        )
        second_layer_aux.train_fc_layer(train_loader)
        most_important_indices, _ = (
            second_layer_aux.pruning()
        )

        return second_layer_aux, most_important_indices

    @staticmethod
    def replace_retrain_second_layer(
            cnn_slfn, main_layer, aux_layer, most_important_indices, least_important_indices,
            weight_attr, train_loader, test_loader
    ):
        best_weight_tensor = getattr(aux_layer, weight_attr).data
        best_weights = best_weight_tensor[:, most_important_indices]
        weight_tensor = getattr(main_layer, weight_attr).data
        weight_tensor[:, least_important_indices] = best_weights

        main_layer.train_fc_layer(train_loader)
        main_layer.test_net(
            base_instance=cnn_slfn,
            data_loader=test_loader,
            layer_weights=[main_layer.extended_beta_weights,
                           main_layer.gamma_weights],
            num_layers=2
        )

        return main_layer

    @staticmethod
    def train_eval_third_layer(cnn_slfn, second_layer, train_loader, test_loader):
        third_layer = (
            ThirdLayer(
                second_layer
            )
        )
        third_layer.train_fc_layer(train_loader)
        third_layer.test_net(
            base_instance=cnn_slfn,
            data_loader=test_loader,
            layer_weights=[third_layer.extended_beta_weights,
                           third_layer.extended_gamma_weights,
                           third_layer.delta_weights],
            num_layers=3
        )

        _, least_important_indices = third_layer.pruning()

        return third_layer, least_important_indices

    @staticmethod
    def train_prune_third_layer_auxiliary(second_layer, train_loader):
        third_layer_aux = (
            ThirdLayer(
                second_layer
            )
        )
        third_layer_aux.train_fc_layer(train_loader)
        most_important_indices, _ = third_layer_aux.pruning()

        return third_layer_aux, most_important_indices

    @staticmethod
    def replace_retrain_third_layer(
            cnn_slfn, main_layer, aux_layer, most_important_indices, least_important_indices,
            weight_attr, train_loader, test_loader
    ):
        best_weight_tensor = getattr(aux_layer, weight_attr).data
        best_weights = best_weight_tensor[:, most_important_indices]
        weight_tensor = getattr(main_layer, weight_attr).data
        weight_tensor[:, least_important_indices] = best_weights

        main_layer.train_fc_layer(train_loader)
        main_layer.test_net(cnn_slfn,
                            test_loader,
                            [main_layer.extended_beta_weights,
                             main_layer.extended_gamma_weights,
                             main_layer.delta_weights],
                            3)

        return main_layer

    def main(self, config):
        train_loader = ray.get(config['train_loader'])
        test_loader = ray.get(config['test_loader'])

        # First layer
        cnn_slfn, least_important_filters = (
            self.train_eval_main_net(
                config,
                train_loader,
                test_loader
            )
        )
        aux_cnn_slfn, most_important_filters = (
            self.train_auxiliary_net(
                config,
                train_loader
            )
        )
        cnn_slfn = (
            self.replace_retrain(
                cnn_slfn, aux_cnn_slfn, least_important_filters, most_important_filters, train_loader, test_loader
            )
        )

        # Second layer
        second_layer, least_important_indices = (
            self.train_eval_second_layer(
                cnn_slfn, train_loader, test_loader
            )
        )
        second_layer_aux, most_important_indices = (
            self.train_prune_second_layer_auxiliary(
                cnn_slfn, train_loader
            )
        )
        second_layer = (
            self.replace_retrain_second_layer(
                cnn_slfn,
                second_layer,
                second_layer_aux,
                most_important_indices,
                least_important_indices,
                "extended_beta_weights",
                train_loader,
                test_loader
            )
        )

        # Third layer
        third_layer, least_important_indices = (
            self.train_eval_third_layer(
                cnn_slfn,
                second_layer,
                train_loader,
                test_loader
            )
        )
        third_layer_aux, most_important_indices = (
            self.train_prune_third_layer_auxiliary(
                second_layer,
                train_loader
            )
        )
        third_layer = self.replace_retrain_third_layer(
            cnn_slfn, third_layer, third_layer_aux, most_important_indices, least_important_indices,
            "extended_gamma_weights",
            train_loader,
            test_loader
        )

        session.report({"accuracy": third_layer.accuracy[-1]})

    def tune_params(self):
        scheduler = (
            ASHAScheduler(
                metric="accuracy",
                mode="max",
                max_t=1,
                grace_period=1,
                reduction_factor=2
            )
        )

        reporter = (
            tune.CLIReporter(
                parameter_columns=["num_filters", "conv_kernel", "pruning_percentage", "rcond"],
                metric_columns=["accuracy", "training_iteration"]
            )
        )

        result = (
            tune.run(
                self.main,
                resources_per_trial={"gpu": 0, "cpu": mp.cpu_count()},
                config=self.settings,
                num_samples=16,
                scheduler=scheduler,
                progress_reporter=reporter,
                storage_path="C:/Users/ricsi/Desktop/conv_slfn_hyper"
            )
        )

        best_trial = result.get_best_trial("accuracy", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        train_eval = TrainEval()
        train_eval.tune_params()
    except KeyboardInterrupt:
        logging.error("Keyboard Interrupt")
