import colorama
import logging
import os

from ray import tune
from ray.air import session

from config.dataset_config import drnn_paths_config
from nn.dev_drnn.base_class_dev_drnn import BaseDevDRNN
from nn.models.layer_selector import LayerFactory
from utils.utils import (save_log_to_txt)


class ParamSearch(BaseDevDRNN):
    def __init__(self):
        super().__init__()

        colorama.init()

        self.save_path = drnn_paths_config(self.cfg.get("dataset_name")).get("dev_drnn").get("hyperparam_tuning")
        self.save_log_file = os.path.join(
            self.save_path,
            f"hyperparam_search_best_results_{self.cfg.get("method")}.txt"
        )

        self.hyperparam_config = {
            "rcond": tune.loguniform(1e-30, 1e-1),
            "penalty_term": tune.loguniform(1e-30, 1e-1),
            "scaling_factor": self.cfg.get("scaling_factor"),
                # tune.uniform(0.05, 0.95),
            "error_threshold": self.cfg.get("error_threshold"),
                #tune.loguniform(1e-16, 1e-4),
            "similarity_threshold":  self.cfg.get("similarity_threshold"),
                # tune.uniform(0.05, 0.95),
            "res_rcond": # self.cfg.get("res_rcond"),
                tune.loguniform(1e-30, 1e-1),
            "neurons": self.cfg.get("exp_neurons"),
            "method": self.cfg.get("method")
        }

    def main(self, config):
        try:
            # First layer
            first_layer_cfg = (
                self.get_network_config(
                    "DevDeepRandomizedNeuralNetworkFirstLayer",
                    config
                )
            )

            self.first_layer = (
                LayerFactory.create(
                    network_type="DevDeepRandomizedNeuralNetworkFirstLayer",
                    network_cfg=first_layer_cfg,
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
                    network_type="DevDeepRandomizedNeuralNetworkSecondLayer",
                    network_cfg=second_layer_cfg
                )
            )

            self.second_layer, _, _ = (
                self.model_training_and_evaluation(
                    model=self.second_layer,
                    weights=[
                        self.second_layer.extended_beta_weights,
                        self.second_layer.gamma_weights
                    ],
                    num_hidden_layers=2,
                    verbose=True
                )
            )

            # Third layer
            third_layer_cfg = (
                self.get_network_config(
                    network_type="DevDeepRandomizedNeuralNetworkThirdLayer",
                    config=config
                )
            )

            third_layer = (
                LayerFactory.create(
                    network_type="DevDeepRandomizedNeuralNetworkThirdLayer",
                    network_cfg=third_layer_cfg
                )
            )

            third_layer, training_metrics, testing_metrics = (
                self.model_training_and_evaluation(
                    model=third_layer,
                    weights=[
                        third_layer.extended_beta_weights,
                        third_layer.extended_gamma_weights,
                        third_layer.delta_weights
                    ],
                    num_hidden_layers=3,
                    verbose=True
                )
            )

            session.report({"accuracy": testing_metrics[0]})
        except Exception as e:
            logging.error(e)
            session.report({"accuracy": 0.0})

    def tune_params(self):
        reporter = tune.CLIReporter(
            parameter_columns=["C_penalty", "scaling_factor"],
            metric_columns=["accuracy", "training_iteration"]
        )

        result = tune.run(
            self.main,
            resources_per_trial={
                "cpu": 8,
                "gpu": 0
            },
            config=self.hyperparam_config,
            num_samples=1000,
            progress_reporter=reporter,
            storage_path=self.save_path
        )

        best_trial = result.get_best_trial("accuracy", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

        save_log_to_txt(
            output_file=self.save_log_file,
            result=result,
            operation="accuracy"
        )

if __name__ == "__main__":
    param_search_instance = ParamSearch()
    param_search_instance.tune_params()