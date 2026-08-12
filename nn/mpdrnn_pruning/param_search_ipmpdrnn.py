import os
import sys
import json
import logging
import gc
import random
import colorama
from datetime import datetime
from typing import Any, Dict, Optional

import optuna
import redis
import ray

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Stopper
from ray.air import session

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import drnn_paths_config
from nn.mpdrnn_pruning.base_class_ipmpdrnn import BaseIPMPDRNN
from nn.models.model_selector import ModelFactory
from utils.utils import save_log_to_txt, setup_logger

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
            return bool(r.get(f"ipmpdrnn:abort:{self.task_id}"))
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


class ParamSearchIPMPDRNN(BaseIPMPDRNN):
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- __I N I T__ ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, override_cfg: Optional[dict] = None, celery_task: Any = None):
        self.celery_task = celery_task
        dataset_name = override_cfg.get("dataset_name") if override_cfg else "connect4"

        try:
            ipmpdrnn_config_path = JSON_FILES_PATHS.get_data_path("config_ipmpdrnn")
            with open(ipmpdrnn_config_path, "r") as f:
                raw_json = json.load(f)
        except Exception:
            raw_json = {"dataset_name": dataset_name}

        raw_json["dataset_name"] = dataset_name

        simple_config = {k: v for k, v in raw_json.items() if not isinstance(v, dict)}
        nested_config = {k: v for k, v in raw_json.items() if isinstance(v, dict)}

        flattened_config = {}
        for key, value in nested_config.items():
            if key != 'hyperparamtuning' and dataset_name in value:
                flattened_config[key] = value[dataset_name]

        full_override_cfg = {**simple_config, **flattened_config}

        full_override_cfg.setdefault("mu", 0)
        full_override_cfg.setdefault("sigma", 0.1)
        full_override_cfg.setdefault("activation", "LeakyReLU")
        full_override_cfg.setdefault("method", "BASE")

        if override_cfg:
            full_override_cfg.update(override_cfg)

        super().__init__(override_cfg=full_override_cfg, celery_task=celery_task)

        # Setup logger and colour
        setup_logger()
        colorama.init()

        self.task_id = self.cfg.get("task_id")
        dataset_name = self.cfg.get("dataset_name")

        h_cfg = self.cfg.get("hyperparamtuning", {}) if isinstance(self.cfg.get("hyperparamtuning"), dict) else {}

        self.rcond_min = float(self.cfg.get("rcond_min", h_cfg.get("rcond", {}).get("from", 1e-30)))
        self.rcond_max = float(self.cfg.get("rcond_max", h_cfg.get("rcond", {}).get("to", 1e-1)))
        self.penalty_min = float(self.cfg.get("penalty_min", h_cfg.get("penalty_term", {}).get("from", 0.1)))
        self.penalty_max = float(self.cfg.get("penalty_max", h_cfg.get("penalty_term", {}).get("to", 30.0)))

        self.sp_min = float(self.cfg.get("sp_min", 0.1))
        self.sp_max = float(self.cfg.get("sp_max", 0.9))
        self.num_aux_min = int(self.cfg.get("num_aux_min", 1))
        self.num_aux_max = int(self.cfg.get("num_aux_max", 5))

        self.l1_min = int(self.cfg.get("l1_min", 600))
        self.l1_max = int(self.cfg.get("l1_max", 1000))
        self.l2_min = int(self.cfg.get("l2_min", 200))
        self.l2_max = int(self.cfg.get("l2_max", 500))
        self.l3_min = int(self.cfg.get("l3_min", 50))
        self.l3_max = int(self.cfg.get("l3_max", 100))

        self.activation = self.cfg.get("activation", "LeakyReLU")
        self.method = self.cfg.get("method", "BASE")
        self.mu = self.cfg.get("mu", 0)
        self.sigma = self.cfg.get("sigma", 0.1)

        self.hyperparam_config: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "task_id": self.task_id,
            "activation": self.activation,
            "method": self.method,
            "rcond": tune.loguniform(self.rcond_min, self.rcond_max),
            "subset_percentage": tune.uniform(self.sp_min, self.sp_max),
            "num_aux_net": tune.randint(self.num_aux_min, self.num_aux_max + 1),
            "neurons_l1": tune.randint(self.l1_min, self.l1_max + 1),
            "neurons_l2": tune.randint(self.l2_min, self.l2_max + 1),
            "neurons_l3": tune.randint(self.l3_min, self.l3_max + 1),
        }

        if self.method == "EXP_ORT_C":
            self.hyperparam_config["penalty_term"] = tune.uniform(self.penalty_min, self.penalty_max)

        self.base_save_path = drnn_paths_config(dataset_name).get("ipmpdrnn").get("hyperparam_tuning")
        os.makedirs(self.base_save_path, exist_ok=True)
        self.current_optuna_save_dir = None

    @staticmethod
    def generate_neurons():
        # Generate 3 unique neurons with large gaps, in descending order
        neurons = sorted(random.sample(range(100, 1000, 300), 3), reverse=True)
        return neurons

    def is_aborted(self) -> bool:
        if not self.task_id:
            return False
        try:
            r = redis.Redis.from_url(REDIS_URL)
            return bool(r.get(f"ipmpdrnn:abort:{self.task_id}"))
        except Exception:
            return False

    def fit(self, config: dict) -> None:
        if self.is_aborted():
            raise RuntimeError("ABORTED_BY_USER")

        if "neurons" not in config and "neurons_l1" in config:
            config["neurons"] = [config["neurons_l1"], config["neurons_l2"], config["neurons_l3"]]

        full_config = {**self.cfg, **config}
        full_config["activation"] = self.activation
        full_config["method"] = self.method
        full_config["mu"] = self.mu
        full_config["sigma"] = self.sigma

        # Create model
        net_cfg = self.get_network_config("MultiPhaseDeepRandomizedNeuralNetworkBase", full_config)
        self.initial_model = ModelFactory.create("MultiPhaseDeepRandomizedNeuralNetworkBase", net_cfg)
        # Train and evaluate the model
        self.initial_model, _, _ = self.model_training_and_evaluation(
            model=self.initial_model,
            eval_set=self.valid_loader,
            weights=self.initial_model.beta_weights,
            num_hidden_layers=1,
            verbose=False
        )
        # Pruning
        least_important_prune_indices = self.prune_initial_model(
            self.initial_model, set_weights_to_zero=False, config=full_config
        )
        # Create aux model 1
        self.initial_model = self.create_train_prune_initial_aux_model(
            model=self.initial_model,
            model_type="MultiPhaseDeepRandomizedNeuralNetworkBase",
            least_important_prune_indices=least_important_prune_indices,
            config=full_config
        )
        self.initial_model, _, _ = self.model_training_and_evaluation(
            model=self.initial_model,
            eval_set=self.valid_loader,
            weights=self.initial_model.beta_weights,
            num_hidden_layers=1,
            verbose=False
        )

        net_cfg = self.get_network_config("MultiPhaseDeepRandomizedNeuralNetworkSubsequent", full_config)
        self.subsequent_model = ModelFactory.create("MultiPhaseDeepRandomizedNeuralNetworkSubsequent", net_cfg)
        self.subsequent_model, _, _ = self.model_training_and_evaluation(
            model=self.subsequent_model,
            eval_set=self.valid_loader,
            weights=[self.subsequent_model.extended_beta_weights, self.subsequent_model.gamma_weights],
            num_hidden_layers=2,
            verbose=False
        )
        least_important_prune_indices = self.prune_subsequent_model(
            model=self.subsequent_model,
            set_weights_to_zero=False,
            config=full_config
        )
        self.subsequent_model = self.create_train_prune_subsequent_aux_model(
            model=self.subsequent_model,
            model_type="MultiPhaseDeepRandomizedNeuralNetworkSubsequent",
            least_important_prune_indices=least_important_prune_indices,
            config=full_config
        )
        self.subsequent_model, _, _ = self.model_training_and_evaluation(
            model=self.subsequent_model,
            eval_set=self.valid_loader,
            weights=[self.subsequent_model.extended_beta_weights, self.subsequent_model.gamma_weights],
            num_hidden_layers=2,
            verbose=False
        )

        # Final Model
        net_cfg = self.get_network_config(network_type="MultiPhaseDeepRandomizedNeuralNetworkFinal", config=full_config)
        final_model = ModelFactory.create("MultiPhaseDeepRandomizedNeuralNetworkFinal", net_cfg)
        final_model, final_model_subs_weights_training_metrics, final_model_subs_weights_testing_metrics = self.model_training_and_evaluation(
            model=final_model,
            eval_set=self.valid_loader,
            weights=[final_model.extended_beta_weights, final_model.extended_gamma_weights, final_model.delta_weights],
            num_hidden_layers=3,
            verbose=False
        )
        least_important_prune_indices = self.prune_final_model(
            model=final_model,
            set_weights_to_zero=False,
            config=full_config
        )
        final_model = self.create_train_prune_final_aux_model(
            model=final_model,
            model_type="MultiPhaseDeepRandomizedNeuralNetworkFinal",
            least_important_prune_indices=least_important_prune_indices,
            config=full_config
        )
        final_model, final_model_model_subs_weights_training_metrics, final_model_model_subs_weights_testing_metrics = self.model_training_and_evaluation(
            model=final_model,
            eval_set=self.valid_loader,
            weights=[final_model.extended_beta_weights, final_model.extended_gamma_weights, final_model.delta_weights],
            num_hidden_layers=3,
            verbose=False
        )

        best = final_model_subs_weights_testing_metrics if (final_model_subs_weights_testing_metrics[0] > final_model_model_subs_weights_testing_metrics[0]) else final_model_model_subs_weights_testing_metrics
        valid_acc = float(best[0])
        session.report({"accuracy": valid_acc})

    def optuna_objective(self, trial: optuna.Trial) -> float:
        if self.is_aborted():
            logging.info(f"🛑 [Task {self.task_id}] ABORT SIGNAL DETECTED. Stopping Optuna study!")
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

        rcond = trial.suggest_float("rcond", self.rcond_min, self.rcond_max, log=True)
        penalty_term = trial.suggest_float("penalty_term", self.penalty_min, self.penalty_max) if self.method == "EXP_ORT_C" else None
        subset_percentage = trial.suggest_float("subset_percentage", self.sp_min, self.sp_max)
        num_aux_net = trial.suggest_int("num_aux_net", self.num_aux_min, self.num_aux_max)

        n1 = trial.suggest_int("neurons_l1", self.l1_min, self.l1_max)
        n2 = trial.suggest_int("neurons_l2", self.l2_min, self.l2_max)
        n3 = trial.suggest_int("neurons_l3", self.l3_min, self.l3_max)

        config = {
            "rcond": rcond,
            "penalty_term": penalty_term,
            "subset_percentage": subset_percentage,
            "num_aux_net": num_aux_net,
            "neurons": [n1, n2, n3]
        }

        try:
            full_config = {**self.cfg, **config}
            full_config["activation"] = self.activation
            full_config["method"] = self.method
            full_config["mu"] = self.mu
            full_config["sigma"] = self.sigma

            # Create model
            net_cfg = self.get_network_config("MultiPhaseDeepRandomizedNeuralNetworkBase", full_config)
            self.initial_model = ModelFactory.create("MultiPhaseDeepRandomizedNeuralNetworkBase", net_cfg)
            self.initial_model, _, _ = self.model_training_and_evaluation(
                model=self.initial_model, eval_set=self.valid_loader, weights=self.initial_model.beta_weights, num_hidden_layers=1, verbose=False
            )
            least_important_prune_indices = self.prune_initial_model(
                self.initial_model, set_weights_to_zero=False, config=full_config
            )
            self.initial_model = self.create_train_prune_initial_aux_model(
                model=self.initial_model, model_type="MultiPhaseDeepRandomizedNeuralNetworkBase", least_important_prune_indices=least_important_prune_indices, config=full_config
            )
            self.initial_model, _, _ = self.model_training_and_evaluation(
                model=self.initial_model, eval_set=self.valid_loader, weights=self.initial_model.beta_weights, num_hidden_layers=1, verbose=False
            )

            net_cfg = self.get_network_config("MultiPhaseDeepRandomizedNeuralNetworkSubsequent", full_config)
            self.subsequent_model = ModelFactory.create("MultiPhaseDeepRandomizedNeuralNetworkSubsequent", net_cfg)
            self.subsequent_model, _, _ = self.model_training_and_evaluation(
                model=self.subsequent_model, eval_set=self.valid_loader, weights=[self.subsequent_model.extended_beta_weights, self.subsequent_model.gamma_weights], num_hidden_layers=2, verbose=False
            )
            least_important_prune_indices = self.prune_subsequent_model(
                model=self.subsequent_model, set_weights_to_zero=False, config=full_config
            )
            self.subsequent_model = self.create_train_prune_subsequent_aux_model(
                model=self.subsequent_model, model_type="MultiPhaseDeepRandomizedNeuralNetworkSubsequent", least_important_prune_indices=least_important_prune_indices, config=full_config
            )
            self.subsequent_model, _, _ = self.model_training_and_evaluation(
                model=self.subsequent_model, eval_set=self.valid_loader, weights=[self.subsequent_model.extended_beta_weights, self.subsequent_model.gamma_weights], num_hidden_layers=2, verbose=False
            )

            net_cfg = self.get_network_config(network_type="MultiPhaseDeepRandomizedNeuralNetworkFinal", config=full_config)
            final_model = ModelFactory.create("MultiPhaseDeepRandomizedNeuralNetworkFinal", net_cfg)
            final_model, final_model_subs_weights_training_metrics, final_model_subs_weights_testing_metrics = self.model_training_and_evaluation(
                model=final_model, eval_set=self.valid_loader, weights=[final_model.extended_beta_weights, final_model.extended_gamma_weights, final_model.delta_weights], num_hidden_layers=3, verbose=False
            )
            least_important_prune_indices = self.prune_final_model(
                model=final_model, set_weights_to_zero=False, config=full_config
            )
            final_model = self.create_train_prune_final_aux_model(
                model=final_model, model_type="MultiPhaseDeepRandomizedNeuralNetworkFinal", least_important_prune_indices=least_important_prune_indices, config=full_config
            )
            final_model, final_model_model_subs_weights_training_metrics, final_model_model_subs_weights_testing_metrics = self.model_training_and_evaluation(
                model=final_model, eval_set=self.valid_loader, weights=[final_model.extended_beta_weights, final_model.extended_gamma_weights, final_model.delta_weights], num_hidden_layers=3, verbose=False
            )

            best = final_model_subs_weights_testing_metrics if (final_model_subs_weights_testing_metrics[0] > final_model_model_subs_weights_testing_metrics[0]) else final_model_model_subs_weights_testing_metrics
            valid_acc = float(best[0])
            trial.report(valid_acc, step=1)

            if self.current_optuna_save_dir:
                trial_dir = os.path.join(self.current_optuna_save_dir, "trials", f"trial_{trial.number:02d}")
                os.makedirs(trial_dir, exist_ok=True)
                params_dict = {
                    "rcond": rcond,
                    "penalty_term": penalty_term,
                    "subset_percentage": subset_percentage,
                    "num_aux_net": num_aux_net,
                    "neurons": [n1, n2, n3]
                }
                with open(os.path.join(trial_dir, "params.json"), "w") as pf:
                    json.dump(params_dict, pf, indent=4)

                results_dict = {
                    "trial_number": trial.number,
                    "best_val_accuracy": valid_acc,
                    "state": "FINISHED"
                }
                with open(os.path.join(trial_dir, "result.json"), "w") as rf:
                    json.dump(results_dict, rf, indent=4)

            return valid_acc

        finally:
            gc.collect()

    def tune_params_optuna(self, n_trials: int = 25) -> dict:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        optuna_save_dir = os.path.join(self.base_save_path, "optuna", timestamp)
        os.makedirs(optuna_save_dir, exist_ok=True)
        self.current_optuna_save_dir = optuna_save_dir

        save_log_file = os.path.join(optuna_save_dir, "optuna_ipmpdrnn_hyperparam_search_best_results.txt")

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
            f.write("=== OPTUNA IPMPDRNN HYPERPARAMETER SEARCH RESULTS ===\n")
            f.write(f"Dataset: {self.cfg.get('dataset_name')}\n")
            f.write(f"Activation: {self.activation}\n")
            f.write(f"Method: {self.method}\n")
            f.write(f"Best Accuracy: {best_value:.4f}\n")
            f.write("Best Parameters:\n")
            for key, value in best_params.items():
                f.write(f"  {key}: {value}\n")

        try:
            df = study.trials_dataframe()
            csv_path = os.path.join(optuna_save_dir, "optuna_ipmpdrnn_trials_history.csv")
            df.to_csv(csv_path, index=False)

            json_path = os.path.join(optuna_save_dir, "optuna_ipmpdrnn_best_params.json")
            with open(json_path, "w") as jf:
                json.dump({"best_accuracy": best_value, "best_params": best_params}, jf, indent=4)
        except Exception as e:
            logging.warning(f"Could not export Optuna CSV/JSON to disk: {e}")

        return {
            "best_accuracy": float(best_value),
            "best_params": best_params,
            "backend": "optuna"
        }

    def tune_params_ray(self, n_trials: int = 25) -> dict:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ray_save_dir = os.path.join(self.base_save_path, "ray", timestamp)
        os.makedirs(ray_save_dir, exist_ok=True)

        save_log_file = os.path.join(ray_save_dir, "ray_ipmpdrnn_hyperparam_search_best_results.txt")

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
            max_t=32,
            grace_period=4,
            reduction_factor=2
        )

        param_cols = ["rcond", "subset_percentage", "num_aux_net", "neurons_l1", "neurons_l2", "neurons_l3"]
        if self.method == "EXP_ORT_C":
            param_cols.insert(1, "penalty_term")

        reporter = tune.CLIReporter(
            parameter_columns=param_cols,
            metric_columns=["accuracy", "training_iteration"]
        )

        stopper = RedisAbortStopper(self.task_id)

        celery_task_backup = self.celery_task
        self.celery_task = None
        progress_callback = RayCeleryProgressCallback(celery_task_backup, n_trials)

        try:
            result = tune.run(
                self.fit,
                stop=stopper,
                resources_per_trial={"cpu": 1, "gpu": 0},
                max_concurrent_trials=1,
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
                if k not in ["dataset_name", "task_id", "activation", "method", "mu", "sigma"]
            }

            return {
                "best_accuracy": float(best_trial.last_result["accuracy"]) if (best_trial and best_trial.last_result) else 0.0,
                "best_params": clean_best_params,
                "backend": "ray"
            }

        except Exception as e:
            if self.is_aborted() or "ABORTED" in str(e):
                raise RuntimeError("ABORTED_BY_USER")
            raise

        finally:
            self.celery_task = celery_task_backup
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
    param_searh = ParamSearchIPMPDRNN()
    param_searh.tune_params()