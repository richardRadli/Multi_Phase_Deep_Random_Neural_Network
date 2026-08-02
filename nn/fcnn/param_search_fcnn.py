import os
import sys
import json
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import redis
import ray

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Stopper
from ray.air import session
from torch.utils.data import DataLoader
from typing import Any, Dict, Optional

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import general_dataset_configs, fcnn_paths_configs
from nn.models.fcnn_model import FullyConnectedNeuralNetwork
from nn.dataloaders.npz_dataloader import NpzDataset
from utils.utils import device_selector, load_config_json, save_log_to_txt

optuna.logging.set_verbosity(optuna.logging.WARNING)
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://redis_broker:6379/1")


class RedisAbortStopper(Stopper):
    def __init__(self, task_id: Optional[str]):
        self.task_id = task_id

    def __call__(self, trial_id: str, result: dict) -> bool:
        return self._is_aborted()

    def stop_all(self) -> bool:
        return self._is_aborted()

    def _is_aborted(self) -> bool:
        if not self.task_id:
            return False
        try:
            r = redis.Redis.from_url(REDIS_URL)
            return bool(r.get(f"fcnn:abort:{self.task_id}"))
        except Exception:
            return False


class RayCeleryProgressCallback(tune.Callback):
    def __init__(self, celery_task: Any, total_trials: int):
        self.celery_task = celery_task
        self.total_trials = total_trials
        self.completed_trials = 0
        self.best_acc = 0.0

    def on_trial_complete(self, iteration: int, trials: list, trial: Any, **info):
        self.completed_trials += 1
        acc = float(trial.last_result.get("accuracy", 0.0)) if trial.last_result else 0.0
        if acc > self.best_acc:
            self.best_acc = acc

        if self.celery_task:
            progress_percent = round((self.completed_trials / self.total_trials) * 100, 1)
            self.celery_task.update_state(
                state="PROGRESS",
                meta={
                    "status": f"Running trial {self.completed_trials}/{self.total_trials}",
                    "progress_percent": progress_percent,
                    "current_trial": self.completed_trials,
                    "total_trials": self.total_trials,
                    "best_accuracy_so_far": self.best_acc
                }
            )


def ray_train_func(config: Dict[str, Any]) -> None:
    dataset_name = config["dataset_name"]
    gen_ds_cfg = general_dataset_configs(dataset_name)
    file_path = gen_ds_cfg.get("cached_dataset_file")

    device = device_selector(preferred_device=config.get("device", "cuda"))
    criterion = nn.CrossEntropyLoss()

    model = FullyConnectedNeuralNetwork(
        input_size=gen_ds_cfg.get("num_features"),
        hidden_size=config["hidden_neurons"],
        output_size=gen_ds_cfg.get("num_classes")
    ).to(device)

    optimizer_name = config.get("optimizer", "adam")
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config.get("momentum", 0.9))
    else:
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    train_dataset = NpzDataset(file_path, operation="train")
    val_dataset = NpzDataset(file_path, operation="valid")
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    task_id = config.get("task_id")
    max_epochs = config.get("max_epochs", 100)

    for epoch in range(max_epochs):
        if task_id:
            try:
                r = redis.Redis.from_url(REDIS_URL)
                if bool(r.get(f"fcnn:abort:{task_id}")):
                    return
            except Exception:
                pass

        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                predicted_labels = torch.argmax(output, 1)
                correct_predictions += (predicted_labels == torch.argmax(target, dim=1)).sum().item()
                total_samples += target.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct_predictions / total_samples

        session.report({"loss": val_loss, "accuracy": val_accuracy})


class HyperparameterSearch:
    def __init__(self, override_cfg: Optional[dict] = None, celery_task: Any = None):
        self.celery_task = celery_task
        self.cfg = load_config_json(
            json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_fcnn"),
            json_filename=JSON_FILES_PATHS.get_data_path("config_fcnn")
        )

        if override_cfg:
            self.cfg.update(override_cfg)

        self.task_id = self.cfg.get("task_id")
        dataset_name = self.cfg.get("dataset_name")

        self.gen_ds_cfg = general_dataset_configs(dataset_name)
        self.file_path = general_dataset_configs(dataset_name).get("cached_dataset_file")

        self.device = device_selector(preferred_device=self.cfg.get("device", "cuda"))
        self.optimizer = self.cfg.get("optimizer", "adam")
        self.criterion = nn.CrossEntropyLoss()
        self.max_epochs = int(self.cfg.get("epochs", 100))

        self.lr_min = float(self.cfg.get("lr_min", 4e-4))
        self.lr_max = float(self.cfg.get("lr_max", 1e-1))
        self.hidden_neurons_options = self.cfg.get("hidden_neurons_options", [216, 500, 866, 1000, 2000])
        self.batch_size_options = self.cfg.get("batch_size_options", [32, 64, 128])
        self.momentum_min = float(self.cfg.get("momentum_min", 0.5))
        self.momentum_max = float(self.cfg.get("momentum_max", 0.99))

        self.hyperparam_config = {
            "dataset_name": dataset_name,
            "device": self.cfg.get("device", "cuda"),
            "optimizer": self.optimizer,
            "task_id": self.task_id,
            "max_epochs": self.max_epochs,
            "lr": tune.loguniform(self.lr_min, self.lr_max),
            "momentum": tune.uniform(self.momentum_min, self.momentum_max),
            "hidden_neurons": tune.grid_search(self.hidden_neurons_options),
            "batch_size": tune.choice(self.batch_size_options)
        }

        self.base_save_path = fcnn_paths_configs(dataset_name).get("hyperparam_tuning")
        os.makedirs(self.base_save_path, exist_ok=True)

    def is_aborted(self) -> bool:
        if not self.task_id:
            return False
        try:
            r = redis.Redis.from_url(REDIS_URL)
            return bool(r.get(f"fcnn:abort:{self.task_id}"))
        except Exception:
            return False

    def fit(self, config: Dict[str, Any]) -> None:
        model = FullyConnectedNeuralNetwork(
            input_size=self.gen_ds_cfg.get("num_features"),
            hidden_size=config["hidden_neurons"],
            output_size=self.gen_ds_cfg.get("num_classes")
        ).to(self.device)

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

        for epoch in range(self.max_epochs):
            if self.is_aborted():
                raise RuntimeError("ABORTED_BY_USER")

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

    def optuna_objective(self, trial: optuna.Trial) -> float:
        if self.is_aborted():
            logging.info(f" [Task {self.task_id}] ABORT SIGNAL DETECTED. Stopping Optuna study!")
            trial.study.stop()
            raise optuna.exceptions.OptunaError("ABORTED_BY_USER")

        total_trials = self.cfg.get("n_trials", 25)
        current_trial = trial.number + 1
        progress_percent = round((current_trial / total_trials) * 100, 1)

        try:
            best_val_so_far = float(trial.study.best_value)
        except ValueError:
            best_val_so_far = 0.0

        if self.celery_task:
            self.celery_task.update_state(
                state="PROGRESS",
                meta={
                    "status": f"Running trial {current_trial}/{total_trials}",
                    "progress_percent": progress_percent,
                    "current_trial": current_trial,
                    "total_trials": total_trials,
                    "best_accuracy_so_far": best_val_so_far
                }
            )

        lr = trial.suggest_float("lr", self.lr_min, self.lr_max, log=True)
        hidden_neurons = trial.suggest_categorical("hidden_neurons", self.hidden_neurons_options)
        batch_size = trial.suggest_categorical("batch_size", self.batch_size_options)
        momentum = trial.suggest_float("momentum", self.momentum_min,
                                       self.momentum_max) if self.optimizer == "sgd" else None

        total_trials = self.cfg.get("n_trials", 25)
        logging.info(f"───  [Trial {trial.number:02d}/{total_trials} START] ────────────────────────────────────────")
        logging.info(f"    Params: lr={lr:.6f} | hidden={hidden_neurons} | batch={batch_size}" + (
            f" | momentum={momentum:.4f}" if momentum else ""))

        model = FullyConnectedNeuralNetwork(
            input_size=self.gen_ds_cfg.get("num_features"),
            hidden_size=hidden_neurons,
            output_size=self.gen_ds_cfg.get("num_classes")
        ).to(self.device)

        if self.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)

        train_dataset = NpzDataset(self.file_path, operation="train")
        val_dataset = NpzDataset(self.file_path, operation="valid")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_accuracy = 0.0

        for epoch in range(self.max_epochs):
            if self.is_aborted():
                logging.info(f" [Task {self.task_id}] Abort signal detected at Trial {trial.number}. Stopping study!")
                trial.study.stop()
                raise optuna.exceptions.OptunaError("ABORTED_BY_USER")

            model.train()
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

            model.eval()
            correct_predictions = 0
            total_samples = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    predicted_labels = torch.argmax(output, 1)
                    correct_predictions += (predicted_labels == torch.argmax(target, dim=1)).sum().item()
                    total_samples += target.size(0)

            val_accuracy = correct_predictions / total_samples
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

            trial.report(val_accuracy, epoch)

            is_last = (epoch + 1 == self.max_epochs)
            if (epoch + 1) % 5 == 0 or epoch == 0 or is_last:
                prefix = "    └──" if is_last else "    ├──"
                logging.info(
                    f"{prefix} Epoch {epoch + 1:02d}/{self.max_epochs:02d} ──► Val Acc: {val_accuracy * 100:.2f}%")

            if trial.should_prune():
                logging.info(
                    f"     [Trial {trial.number:02d} PRUNED] at epoch {epoch + 1}/{self.max_epochs} (Acc: {val_accuracy * 100:.2f}%)")

                trial_dir = os.path.join(self.base_save_path, "optuna_runs", "trials", f"trial_{trial.number:02d}")
                os.makedirs(trial_dir, exist_ok=True)
                with open(os.path.join(trial_dir, "result.json"), "w") as rf:
                    json.dump({"trial_number": trial.number, "best_val_accuracy": float(best_val_accuracy),
                               "state": "PRUNED"}, rf, indent=4)

                raise optuna.exceptions.TrialPruned()

        try:
            study_best = trial.study.best_value
            study_best_num = trial.study.best_trial.number
            if best_val_accuracy > study_best:
                study_best = best_val_accuracy
                study_best_num = trial.number
        except ValueError:
            study_best = best_val_accuracy
            study_best_num = trial.number

        logging.info("┌──────────────────────────────────────────────────────────┐")
        logging.info(f"│ 🏁 TRIAL {trial.number:02d} FINISHED | Score: {best_val_accuracy * 100:.2f}%".ljust(59) + "│")
        logging.info(f"│ 🏆 ALL-TIME BEST:    {study_best * 100:.2f}% (Trial {study_best_num:02d})".ljust(59) + "│")
        logging.info("└──────────────────────────────────────────────────────────┘")

        trial_dir = os.path.join(self.base_save_path, "optuna_runs", "trials", f"trial_{trial.number:02d}")
        os.makedirs(trial_dir, exist_ok=True)

        params_dict = {
            "lr": lr,
            "hidden_neurons": hidden_neurons,
            "batch_size": batch_size,
            "momentum": momentum
        }
        with open(os.path.join(trial_dir, "params.json"), "w") as pf:
            json.dump(params_dict, pf, indent=4)

        results_dict = {
            "trial_number": trial.number,
            "best_val_accuracy": float(best_val_accuracy),
            "state": "FINISHED"
        }
        with open(os.path.join(trial_dir, "result.json"), "w") as rf:
            json.dump(results_dict, rf, indent=4)

        return best_val_accuracy

    def tune_params_optuna(self, n_trials: int = 25) -> dict:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        optuna_save_dir = os.path.join(self.base_save_path, "optuna", timestamp)
        os.makedirs(optuna_save_dir, exist_ok=True)

        save_log_file = os.path.join(optuna_save_dir, f"optuna_{self.optimizer}_hyperparam_search_best_results.txt")

        logging.info("==================================================")
        logging.info("         OPTUNA SEARCH CONFIGURATION SUMMARY      ")
        logging.info("==================================================")
        logging.info(f"  Target Dataset:       {self.cfg.get('dataset_name')}")
        logging.info(f"  Task ID:              {self.task_id}")
        logging.info(f"  Optimizer:            {self.optimizer}")
        logging.info(f"  Total Trials:         {n_trials}")
        logging.info(f"  Max Epochs / Trial:   {self.max_epochs}")
        logging.info(f"  Patience (Pruning):   {self.cfg.get('patience', 5)}")
        logging.info(f"  Learning Rate Range:  [{self.lr_min:.6f} -> {self.lr_max:.6f}]")
        logging.info(f"  Hidden Neurons Opt:   {self.hidden_neurons_options}")
        logging.info(f"  Batch Size Options:   {self.batch_size_options}")
        if self.optimizer == "sgd":
            logging.info(f"  Momentum Range:       [{self.momentum_min} -> {self.momentum_max}]")
        logging.info("==================================================")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=1234 if self.cfg.get("seed") else None),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=3,
                n_warmup_steps=self.cfg.get("patience", 5)
            )
        )

        try:
            study.optimize(self.optuna_objective, n_trials=n_trials)
        except optuna.exceptions.OptunaError as oe:
            if "ABORTED_BY_USER" in str(oe):
                raise RuntimeError("ABORTED_BY_USER")
            raise oe

        if self.is_aborted():
            raise RuntimeError("ABORTED_BY_USER")

        best_value = study.best_value if study.best_trial else 0.0
        best_params = study.best_params if study.best_trial else {}

        with open(save_log_file, "w") as f:
            f.write("=== OPTUNA HYPERPARAMETER SEARCH RESULTS ===\n")
            f.write(f"Dataset: {self.cfg.get('dataset_name')}\n")
            f.write(f"Optimizer: {self.optimizer}\n")
            f.write(f"Best Accuracy: {best_value:.4f}\n")
            f.write("Best Parameters:\n")
            for key, value in best_params.items():
                f.write(f"  {key}: {value}\n")

        try:
            df = study.trials_dataframe()
            csv_path = os.path.join(optuna_save_dir, f"optuna_{self.optimizer}_trials_history.csv")
            df.to_csv(csv_path, index=False)

            json_path = os.path.join(optuna_save_dir, f"optuna_{self.optimizer}_best_params.json")
            with open(json_path, "w") as jf:
                json.dump({"best_accuracy": best_value, "best_params": best_params}, jf, indent=4)
        except Exception as e:
            logging.warning(f"Could not export Optuna CSV/JSON to disk: {e}")

        return {
            "best_accuracy": float(best_value),
            "best_params": best_params
        }

    def tune_params_ray(self, n_trials: int = 25) -> dict:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ray_save_dir = os.path.join(self.base_save_path, "ray", timestamp)
        os.makedirs(ray_save_dir, exist_ok=True)

        save_log_file = os.path.join(ray_save_dir, f"ray_{self.optimizer}_hyperparam_search_best_results.txt")

        if not hasattr(sys.stderr, "fileno"):
            sys.stderr.fileno = lambda: 2
        if not hasattr(sys.stdout, "fileno"):
            sys.stdout.fileno = lambda: 1

        if not ray.is_initialized():
            ray.init(
                num_cpus=6,
                ignore_reinit_error=True,
                include_dashboard=False,
                configure_logging=False,
            )

        scheduler = ASHAScheduler(
            metric="accuracy",
            mode="max",
            max_t=self.max_epochs,
            grace_period=self.cfg.get("patience", 10),
            reduction_factor=2
        )

        reporter = tune.CLIReporter(
            parameter_columns=["lr", "momentum", "hidden_neurons", "batch_size"],
            metric_columns=["loss", "accuracy", "training_iteration"]
        )

        stopper = RedisAbortStopper(self.task_id)
        progress_callback = RayCeleryProgressCallback(self.celery_task, n_trials)

        try:
            result = tune.run(
                ray_train_func,
                stop=stopper,
                resources_per_trial={"cpu": 6, "gpu": 0},
                config=self.hyperparam_config,
                num_samples=n_trials,
                scheduler=scheduler,
                progress_reporter=reporter,
                verbose=1,
                storage_path=ray_save_dir,
                name="",
                trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
                callbacks=[progress_callback]
            )

            if self.is_aborted():
                raise RuntimeError("ABORTED_BY_USER")

            save_log_to_txt(
                output_file=save_log_file,
                result=result,
                operation="accuracy"
            )

            best_trial = result.get_best_trial("accuracy", "max", "last")
            raw_best_params = best_trial.config if best_trial else {}

            clean_best_params = {
                k: v for k, v in raw_best_params.items()
                if k not in ["dataset_name", "task_id", "device", "optimizer", "max_epochs"]
            }

            return {
                "best_accuracy": float(best_trial.last_result["accuracy"]) if (
                            best_trial and best_trial.last_result) else 0.0,
                "best_params": clean_best_params
            }

        except Exception as e:
            if self.is_aborted() or "ABORTED" in str(e):
                raise RuntimeError("ABORTED_BY_USER")
            raise

        finally:
            if ray.is_initialized():
                ray.shutdown()

    def tune_params(self, backend: str = "optuna", n_trials: int = 25) -> dict:
        if backend.lower() == "optuna":
            return self.tune_params_optuna(n_trials=n_trials)
        elif backend.lower() == "ray":
            return self.tune_params_ray(n_trials=n_trials)
        else:
            raise ValueError(f"Unsupported backend '{backend}'. Choose 'optuna' or 'ray'.")


if __name__ == '__main__':
    try:
        hyper_par_tune = HyperparameterSearch()
        hyper_par_tune.tune_params(backend="optuna", n_trials=25)
    except KeyboardInterrupt as kie:
        logging.error(f'{kie}')