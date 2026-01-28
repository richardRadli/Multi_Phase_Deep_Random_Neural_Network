import random

import colorama
import os

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session

from config.dataset_config import drnn_paths_config
from nn.dev_drnn.base_class_dev_drnn import BaseDevDRNN
from nn.models.layer_selector import LayerFactory
from utils.utils import (save_log_to_txt)


class ParamSearch(BaseDevDRNN):
    def __init__(self):
        super().__init__()

        colorama.init()

        self.hyperparam_config = {
            "rcond": tune.loguniform(1e-1, 1e-30),
            "penalty_term": tune.uniform(0.1, 30),
            "neurons": self.cfg.get("exp_neurons")
        }

    def main(self, config):
        # First layer
        first_layer_cfg = (
            self.get_network_config(
                "DevDeepRandomizedNeuralNetworkFirstLayer",
                config
            )
        )

        self.first_layer = (
            LayerFactory.create(
                "DevDeepRandomizedNeuralNetworkFirstLayer",
                first_layer_cfg,
                train_loader=self.train_loader
            )
        )

        self.first_layer, _, _ = (
            self.model_training_and_evaluation(
                model=self.first_layer,
                weights=self.first_layer.beta_weights,
                num_hidden_layers=1,
                verbose=True
            )
        )

        # Second layer
        second_layer_cfg = (
            self.get_network_config(
                "DevDeepRandomizedNeuralNetworkSecondLayer",
                config
            )
        )
        self.second_layer = (
            LayerFactory.create(
                "DevDeepRandomizedNeuralNetworkSecondLayer", second_layer_cfg
            )
        )

        self.second_layer, _, _ = (
            self.model_training_and_evaluation(
                model=self.second_layer,
                weights=[self.second_layer.extended_beta_weights,
                         self.second_layer.gamma_weights],
                num_hidden_layers=2,
                verbose=True
            )
        )

        # Third layer
        third_layer_cfg = (
            self.get_network_config(
                network_type="DevDeepRandomizedNeuralNetworkThirdLayer", config=config
            )
        )

        third_layer = (
            LayerFactory.create(
                "DevDeepRandomizedNeuralNetworkThirdLayer", third_layer_cfg
            )
        )

        third_layer, training_metrics, testing_metrics = (
            self.model_training_and_evaluation(
                model=third_layer,
                weights=[third_layer.extended_beta_weights,
                         third_layer.extended_gamma_weights,
                         third_layer.delta_weights],
                num_hidden_layers=3,
                verbose=True
            )
        )

        session.report({"accuracy": testing_metrics[0]})

    def tune_params(self):
        scheduler = (
            ASHAScheduler(
                metric="accuracy",
                mode="max",
                max_t=30,
                grace_period=4,
                reduction_factor=2
            )
        )

        reporter = tune.CLIReporter(
            parameter_columns=["C_penalty", "scaling_factor"],
            metric_columns=["accuracy", "training_iteration"]
        )

        result = tune.run(
            self.main,
            resources_per_trial={
                "cpu": 6,
                "gpu": 0
            },
            config=self.hyperparam_config,
            num_samples=60,
            scheduler=scheduler,
            progress_reporter=reporter,
            storage_path="/home/ricsi/Desktop/hyperparam_search_best_results.txt"
        )

        best_trial = result.get_best_trial("accuracy", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

        # save_log_to_txt(
        #     output_file="",
        #     result=result,
        #     operation="accuracy"
        # )

if __name__ == "__main__":
    param_search_instance = ParamSearch()
    param_search_instance.tune_params()