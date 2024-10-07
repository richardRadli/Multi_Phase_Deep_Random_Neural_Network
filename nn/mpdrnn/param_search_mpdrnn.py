import colorama
import os

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session

from config.json_config import json_config_selector
from config.dataset_config import general_dataset_configs, drnn_paths_config
from nn.models.model_selector import ModelFactory
from utils.utils import (create_train_valid_test_datasets, load_config_json, get_num_of_neurons, save_log_to_txt)


class ParamSearchMPDRNN:
    def __init__(self):
        # Initialize paths and settings
        self.cfg = (
            load_config_json(
                json_schema_filename=json_config_selector("mpdrnn").get("schema"),
                json_filename=json_config_selector("mpdrnn").get("config")
            )
        )

        # Boilerplate
        self.dataset_name = self.cfg.get("dataset_name")
        self.method = self.cfg.get('method')
        self.activation = self.cfg.get('activation')
        self.gen_ds_cfg = general_dataset_configs(self.cfg.get("dataset_name"))

        self.initial_model = None
        self.subsequent_model = None

        file_path = general_dataset_configs(self.dataset_name).get("cached_dataset_file")
        self.train_loader, self.valid_loader, _ = create_train_valid_test_datasets(file_path)

        self.save_path = drnn_paths_config(self.cfg.get("dataset_name")).get("mpdrnn").get("hyperparam_tuning")
        self.save_log_file = os.path.join(self.save_path, "hyperparam_search_best_results.txt")

        self.hyperparam_config = {
            "rcond": tune.loguniform(1e-1, 1e-30),
            "penalty_term": tune.uniform(0.1, 30)
        }
        colorama.init()

    def get_network_config(self, network_type, config):
        net_cfg = {
            "MultiPhaseDeepRandomizedNeuralNetworkBase": {
                "first_layer_num_data": self.gen_ds_cfg.get("num_train_data"),
                "first_layer_num_features": self.gen_ds_cfg.get("num_features"),
                "list_of_hidden_neurons": get_num_of_neurons(self.cfg, self.method),
                "first_layer_output_nodes": self.gen_ds_cfg.get("num_classes"),
                "activation": self.activation,
                "method": self.method,
                "rcond": config.get("rcond"),
                "penalty_term": config.get("penalty_term"),
            },
            "MultiPhaseDeepRandomizedNeuralNetworkSubsequent": {
                "initial_model": self.initial_model,
                "sigma": self.cfg.get('sigma'),
                "mu": self.cfg.get('mu')
            },
            "MultiPhaseDeepRandomizedNeuralNetworkFinal": {
                "subsequent_model": self.subsequent_model,
                "sigma": self.cfg.get('sigma'),
                "mu": self.cfg.get('mu')
            }
        }

        return net_cfg[network_type]

    def model_training_and_evaluation(self, model, weights, num_hidden_layers, verbose):
        model.train_layer(self.train_loader)

        training_metrics = model.predict_and_evaluate(
            dataloader=self.train_loader,
            operation="train",
            layer_weights=weights,
            num_hidden_layers=num_hidden_layers,
            verbose=verbose
        )

        valid_metrics = model.predict_and_evaluate(
            dataloader=self.valid_loader,
            operation="test",
            layer_weights=weights,
            num_hidden_layers=num_hidden_layers,
            verbose=verbose
        )

        return model, training_metrics, valid_metrics

    def main(self, config):
        net_cfg = self.get_network_config("MultiPhaseDeepRandomizedNeuralNetworkBase", config)
        self.initial_model = ModelFactory.create("MultiPhaseDeepRandomizedNeuralNetworkBase", net_cfg)

        self.initial_model, initial_model_training_metrics, initial_model_testing_metrics = (
            self.model_training_and_evaluation(model=self.initial_model,
                                               weights=self.initial_model.beta_weights,
                                               num_hidden_layers=1,
                                               verbose=True)
        )

        # Subsequent Model
        net_cfg = self.get_network_config("MultiPhaseDeepRandomizedNeuralNetworkSubsequent", config)
        self.subsequent_model = ModelFactory.create("MultiPhaseDeepRandomizedNeuralNetworkSubsequent", net_cfg)

        self.subsequent_model, subsequent_model_training_metrics, subsequent_model_testing_metrics = (
            self.model_training_and_evaluation(model=self.subsequent_model,
                                               weights=[self.subsequent_model.extended_beta_weights,
                                                        self.subsequent_model.gamma_weights],
                                               num_hidden_layers=2,
                                               verbose=True)
        )

        net_cfg = self.get_network_config(network_type="MultiPhaseDeepRandomizedNeuralNetworkFinal", config=config)
        final_model = ModelFactory.create("MultiPhaseDeepRandomizedNeuralNetworkFinal", net_cfg)

        final_model, final_model_training_metrics, final_model_testing_metrics = (
            self.model_training_and_evaluation(model=final_model,
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
