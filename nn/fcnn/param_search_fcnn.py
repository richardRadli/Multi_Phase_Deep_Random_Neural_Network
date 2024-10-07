import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from torch.utils.data import DataLoader
from typing import Any, Dict

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import general_dataset_configs, fcnn_paths_configs
from nn.models.fcnn_model import FullyConnectedNeuralNetwork
from nn.dataloaders.npz_dataloader import NpzDataset
from utils.utils import device_selector, load_config_json, save_log_to_txt


class HyperparameterSearch:
    def __init__(self):
        self.cfg = (
            load_config_json(json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_fcnn"),
                             json_filename=JSON_FILES_PATHS.get_data_path("config_fcnn"))
        )

        self.gen_ds_cfg = (
            general_dataset_configs(self.cfg.get("dataset_name"))
        )

        self.file_path = (
            general_dataset_configs(self.cfg.get('dataset_name')).get("cached_dataset_file")
        )

        self.device = device_selector(preferred_device="cuda")

        dataset_name = self.cfg.get("dataset_name")
        self.optimizer = self.cfg.get("optimizer")
        optimization = self.cfg.get("optimization")
        self.learning_rate = optimization.get(self.optimizer).get("learning_rate").get(dataset_name)
        self.momentum = optimization.get(self.optimizer).get("momentum").get(dataset_name) \
            if self.optimizer == "sgd" else None

        self.criterion = nn.CrossEntropyLoss()

        self.hyperparam_config = {
            "lr": tune.loguniform(4e-4, 1e-1),
            "momentum": tune.uniform(0.5, 0.99),
            "hidden_neurons": tune.grid_search([216, 500, 866, 1000, 2000]),
            "batch_size": tune.choice([32, 64, 128])
        }

        self.save_path = fcnn_paths_configs(self.cfg.get("dataset_name")).get("hyperparam_tuning")
        self.save_log_file = os.path.join(self.save_path, f"{self.optimizer}_hyperparam_search_best_results.txt")

    def fit(self, config: Dict[str, Any]) -> None:
        """
         Trains the FCNN model and evaluates it on the validation set.

         Args:
             config (Dict[str, Any]): Configuration dictionary containing hyperparameters for training.
                 Expected keys include:
                 - "hidden_size": Number of hidden units in the model.
                 - "lr": Learning rate for the optimizer.
                 - "batch_size": Size of the training and validation batches.

         Returns:
             None: The function performs training, evaluation, and logs the results but does not return any value.
         """

        model = FullyConnectedNeuralNetwork(input_size=self.gen_ds_cfg.get("num_features"),
                                            hidden_size=config["hidden_neurons"],
                                            output_size=self.gen_ds_cfg.get("num_classes")).to(self.device)

        # Define optimizer
        if self.optimizer not in ["sgd", "adam"]:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        if self.optimizer == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=config["lr"],
                momentum=config["momentum"],
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=config["lr"],
            )

        train_dataset = NpzDataset(self.file_path, operation="train")
        val_dataset = NpzDataset(self.file_path, operation="valid")
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

        for epoch in range(self.cfg.get("epochs")):
            # Training
            model.train()
            train_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            logging.info(f"Train loss: {train_loss:.4f}")

            # Validation
            model.eval()
            val_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = self.criterion(output, target)
                    val_loss += loss.item()
                    predicted_labels = torch.argmax(output, 1)
                    correct_predictions += (predicted_labels == torch.argmax(target, dim=1)).sum().item()
                    total_samples += target.size(0)

            val_loss /= len(val_loader)
            val_accuracy = correct_predictions / total_samples

            session.report({"loss": val_loss, "accuracy": val_accuracy})
    
    def tune_params(self):
        """
        Tunes hyperparameters using Ray Tune and saves the results to a text file.

        This method performs hyperparameter optimization by running multiple trials with different configurations.
        It uses the ASHAScheduler for early stopping and CLIReporter to track progress.

        Returns:
            None: The function does not return any value but performs hyperparameter tuning and logs the results.
        """

        scheduler = ASHAScheduler(
            metric="accuracy",
            mode="max",
            max_t=self.cfg.get("epochs"),
            grace_period=10,
            reduction_factor=2
        )

        reporter = tune.CLIReporter(
            parameter_columns=["lr", "momentum", "hidden_neurons", "batch_size"],
            metric_columns=["loss", "accuracy", "training_iteration"]
        )

        result = tune.run(
            self.fit,
            resources_per_trial={"cpu": 6, "gpu": 1},
            config=self.hyperparam_config,
            num_samples=25,
            scheduler=scheduler,
            progress_reporter=reporter,
            storage_path=self.save_path
        )

        save_log_to_txt(output_file=self.save_log_file,
                        result=result,
                        operation="accuracy")


if __name__ == '__main__':
    try:
        hyper_par_tune = HyperparameterSearch()
        hyper_par_tune.tune_params()
    except KeyboardInterrupt as kie:
        logging.error(f'{kie}')
