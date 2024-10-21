import os

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session

from config.dataset_config import helm_paths_config
from nn.helm.base_class_helm import HELMBase
from utils.utils import save_log_to_txt


class HyperparameterSearchHELM(HELMBase):
    def __init__(self):
        """
        Initializes the HyperparameterSearchHELM class with hyperparameter tuning configuration and paths.

        Returns:
            None
        """

        super().__init__()

        self.hyperparam_config = {
            "C_penalty": tune.loguniform(self.cfg.get("hyperparamtuning").get("C_penalty").get("from"),
                                         self.cfg.get("hyperparamtuning").get("C_penalty").get("to")),
            "scaling_factor": tune.uniform(self.cfg.get("hyperparamtuning").get("scaling_factor").get("from"),
                                           self.cfg.get("hyperparamtuning").get("scaling_factor").get("to"))
        }

        self.save_path = helm_paths_config(self.cfg.get("dataset_name")).get("hyperparam_tuning")
        self.save_log_file = os.path.join(self.save_path, "hyperparam_search_best_results.txt")

    def fit(self, config: dict) -> None:
        """
        Trains the model using the given hyperparameter configuration and evaluates it on the validation set.

        Args:
            config: A dictionary of hyperparameters for the training process.

        Returns:
            None
        """

        t3, beta, beta1, beta2, l3, ps1, ps2 = self.train(config)
        valid_metrics = self.evaluation(beta, beta1, beta2, l3, ps1, ps2, self.valid_loader)
        valid_acc = valid_metrics[0]

        session.report({"accuracy": valid_acc})

    def tune_params(self) -> None:
        """
        Performs hyperparameter tuning using Ray Tune with the specified configuration.

        Returns:
            None
        """

        scheduler = ASHAScheduler(
            metric=self.cfg.get('hyperparamtuning').get("metric"),
            mode=self.cfg.get('hyperparamtuning').get("mode"),
            max_t=self.cfg.get('hyperparamtuning').get('max_t'),
            grace_period=self.cfg.get('hyperparamtuning').get('grace_period'),
            reduction_factor=self.cfg.get('hyperparamtuning').get("reduction_factor")
        )

        reporter = tune.CLIReporter(
            parameter_columns=["C_penalty", "scaling_factor"],
            metric_columns=["accuracy", "training_iteration"]
        )

        result = tune.run(
            self.fit,
            resources_per_trial={"cpu": self.cfg.get("hyperparamtuning").get("num_resources").get("cpu"),
                                 "gpu": self.cfg.get("hyperparamtuning").get("num_resources").get("gpu")},
            config=self.hyperparam_config,
            num_samples=self.cfg.get("hyperparamtuning").get("num_samples"),
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
    try:
        hyper_par_tune = HyperparameterSearchHELM()
        hyper_par_tune.tune_params()
    except KeyboardInterrupt as kie:
        print(kie)
