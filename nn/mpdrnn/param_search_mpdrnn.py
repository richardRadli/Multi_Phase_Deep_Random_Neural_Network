import random

import colorama
import os

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session

from config.dataset_config import drnn_paths_config
from nn.mpdrnn.base_class_mpdrnn import BaseMPDRNN
from nn.models.model_selector import ModelFactory
from utils.utils import (save_log_to_txt)


class ParamSearchMPDRNN(BaseMPDRNN):
    def __init__(self):
        super().__init__()

        colorama.init()

        self.save_path = drnn_paths_config(self.cfg.get("dataset_name")).get("mpdrnn").get("hyperparam_tuning")
        self.save_log_file = os.path.join(self.save_path, "hyperparam_search_best_results.txt")

        self.hyperparam_config = {
            "rcond": tune.loguniform(1e-1, 1e-30),
            "penalty_term": tune.uniform(0.1, 30),
            "neurons": [tune.randint(600, 1000), tune.randint(200, 500), tune.randint(50, 100)],
        }

    @staticmethod
    def generate_neurons():
        return sorted(random.sample(range(100, 1200, 100), 3), reverse=True)

    def main(self, config):
        net_cfg = (
            self.get_network_config(
                "MultiPhaseDeepRandomizedNeuralNetworkBase",
                config
            )
        )

        self.initial_model = (
            ModelFactory.create(
                "MultiPhaseDeepRandomizedNeuralNetworkBase",
                net_cfg
            )
        )

        self.initial_model, initial_model_training_metrics, initial_model_testing_metrics = (
            self.model_training_and_evaluation(
                model=self.initial_model,
                eval_set=self.valid_loader,
                weights=self.initial_model.beta_weights,
                num_hidden_layers=1,
                verbose=True
            )
        )

        # Subsequent Model
        net_cfg = (
            self.get_network_config(
                "MultiPhaseDeepRandomizedNeuralNetworkSubsequent",
                config
            )
        )
        self.subsequent_model = (
            ModelFactory.create(
                "MultiPhaseDeepRandomizedNeuralNetworkSubsequent", net_cfg
            )
        )

        self.subsequent_model, subsequent_model_training_metrics, subsequent_model_testing_metrics = (
            self.model_training_and_evaluation(
                model=self.subsequent_model,
                eval_set=self.valid_loader,
                weights=[self.subsequent_model.extended_beta_weights,
                         self.subsequent_model.gamma_weights],
                num_hidden_layers=2,
                verbose=True
            )
        )

        net_cfg = (
            self.get_network_config(
                network_type="MultiPhaseDeepRandomizedNeuralNetworkFinal", config=config
            )
        )

        final_model = (
            ModelFactory.create(
                "MultiPhaseDeepRandomizedNeuralNetworkFinal", net_cfg
            )
        )

        final_model, final_model_training_metrics, final_model_testing_metrics = (
            self.model_training_and_evaluation(model=final_model,
                                               eval_set=self.valid_loader,
                                               weights=[final_model.extended_beta_weights,
                                                        final_model.extended_gamma_weights,
                                                        final_model.delta_weights],
                                               num_hidden_layers=3,
                                               verbose=True)
        )

        session.report({"accuracy": final_model_testing_metrics[0]})

    def tune_params(self):
        scheduler = ASHAScheduler(
            metric="accuracy",
            mode="max",
            max_t=30,
            grace_period=4,
            reduction_factor=2
        )

        reporter = tune.CLIReporter(
            parameter_columns=["C_penalty", "scaling_factor"],
            metric_columns=["accuracy", "training_iteration"]
        )

        result = tune.run(
            self.main,
            resources_per_trial={"cpu": 6,
                                 "gpu": 0},
            config=self.hyperparam_config,
            num_samples=60,
            scheduler=scheduler,
            progress_reporter=reporter,
            storage_path=self.save_path
        )

        best_trial = result.get_best_trial("accuracy", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

        save_log_to_txt(output_file=self.save_log_file,
                        result=result,
                        operation="accuracy")


if __name__ == "__main__":
    mpdrnn = ParamSearchMPDRNN()
    mpdrnn.tune_params()
