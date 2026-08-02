import os
import sys
import json
import logging
import gc
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
from config.dataset_config import helm_paths_config
from nn.helm.base_class_helm import HELMBase
from utils.utils import save_log_to_txt

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
            return bool(r.get(f"helm:abort:{self.task_id}"))
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


class HyperparameterSearchHELM(HELMBase):
    def __init__(self, override_cfg: Optional[dict] = None, celery_task: Any = None):
        """
        Initializes the HyperparameterSearchHELM class with hyperparameter tuning configuration and paths.

        Returns:
            None
        """

        self.celery_task = celery_task
        dataset_name = override_cfg.get("dataset_name") if override_cfg else "connect4"

        helm_config_path = JSON_FILES_PATHS.get_data_path("config_helm")
        with open(helm_config_path, "r") as f:
            raw_json = json.load(f)

        raw_json["dataset_name"] = dataset_name

        simple_config = {k: v for k, v in raw_json.items() if not isinstance(v, dict)}
        nested_config = {k: v for k, v in raw_json.items() if isinstance(v, dict)}

        flattened_config = {}
        for key, value in nested_config.items():
            if key != 'hyperparamtuning' and dataset_name in value:
                flattened_config[key] = value[dataset_name]

        full_override_cfg = {**simple_config, **flattened_config}

        if not full_override_cfg.get("hidden_neurons"):
            full_override_cfg["hidden_neurons"] = raw_json.get("hidden_neurons", {}).get(dataset_name, [100, 50, 25])

        if override_cfg:
            full_override_cfg.update(override_cfg)

        super().__init__(override_cfg=full_override_cfg, celery_task=celery_task)

        if override_cfg:
            self.cfg.update(override_cfg)

        self.task_id = self.cfg.get("task_id")
        dataset_name = self.cfg.get("dataset_name")

        h_cfg = self.cfg.get("hyperparamtuning", {}) if isinstance(self.cfg.get("hyperparamtuning"), dict) else {}

        self.c_min = float(self.cfg.get("c_min", h_cfg.get("C_penalty", {}).get("from", 1e-30)))
        self.c_max = float(self.cfg.get("c_max", h_cfg.get("C_penalty", {}).get("to", 1e-5)))
        self.scaling_min = float(self.cfg.get("scaling_min", h_cfg.get("scaling_factor", {}).get("from", 0.0)))
        self.scaling_max = float(self.cfg.get("scaling_max", h_cfg.get("scaling_factor", {}).get("to", 1.0)))

        self.hyperparam_config = {
            "dataset_name": dataset_name,
            "task_id": self.task_id,
            "C_penalty": tune.loguniform(self.c_min, self.c_max),
            "scaling_factor": tune.uniform(self.scaling_min, self.scaling_max)
        }

        self.base_save_path = helm_paths_config(dataset_name).get("hyperparam_tuning")
        os.makedirs(self.base_save_path, exist_ok=True)
        self.current_optuna_save_dir = None

    def is_aborted(self) -> bool:
        if not self.task_id:
            return False
        try:
            r = redis.Redis.from_url(REDIS_URL)
            return bool(r.get(f"helm:abort:{self.task_id}"))
        except Exception:
            return False

    def fit(self, config: dict) -> None:
        """
        Trains the model using the given hyperparameter configuration and evaluates it on the validation set.

        Args:
            config: A dictionary of hyperparameters for the training process.

        Returns:
            None
        """

        if self.is_aborted():
            raise RuntimeError("ABORTED_BY_USER")

        full_config = {**self.cfg, **config}
        if "C_penalty" in full_config:
            full_config["penalty"] = full_config["C_penalty"]

        t3, beta, beta1, beta2, l3, ps1, ps2 = self.train(full_config)

        self.beta_l1 = beta1
        self.beta_l2 = beta2
        self.beta = beta

        valid_metrics = self.evaluation(beta, beta1, beta2, l3, ps1, ps2, self.valid_loader)
        valid_acc = float(valid_metrics[0])

        session.report({"accuracy": valid_acc})

    def optuna_objective(self, trial: optuna.Trial) -> float:
        if self.is_aborted():
            logging.info(f" [Task {self.task_id}] ABORT SIGNAL DETECTED. Stopping Optuna study!")
            trial.study.stop()
            raise optuna.exceptions.OptunaError("ABORTED_BY_USER")

        total_trials = self.cfg.get("n_trials", self.cfg.get("hyperparamtuning", {}).get("num_samples", 25))
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

        c_penalty = trial.suggest_float("C_penalty", self.c_min, self.c_max, log=True)
        scaling_factor = trial.suggest_float("scaling_factor", self.scaling_min, self.scaling_max)

        logging.info(f"─── ▶️ [Trial {trial.number:02d}/{total_trials} START] ────────────────────────────────────────")
        logging.info(f"    Params: C_penalty={c_penalty:.6e} | scaling_factor={scaling_factor:.6f}")

        config = {
            **self.cfg,
            "C_penalty": c_penalty,
            "penalty": c_penalty,
            "scaling_factor": scaling_factor
        }

        try:
            t3, beta, beta1, beta2, l3, ps1, ps2 = self.train(config)

            self.beta_l1 = beta1
            self.beta_l2 = beta2
            self.beta = beta

            valid_metrics = self.evaluation(beta, beta1, beta2, l3, ps1, ps2, self.valid_loader)
            valid_acc = float(valid_metrics[0])

            trial.report(valid_acc, step=1)

            try:
                study_best = trial.study.best_value
                study_best_num = trial.study.best_trial.number
                if valid_acc > study_best:
                    study_best = valid_acc
                    study_best_num = trial.number
            except ValueError:
                study_best = valid_acc
                study_best_num = trial.number

            logging.info("┌──────────────────────────────────────────────────────────┐")
            logging.info(f"│ 🏁 TRIAL {trial.number:02d} FINISHED | Score: {valid_acc * 100:.2f}%".ljust(59) + "│")
            logging.info(f"│ 🏆 ALL-TIME BEST:    {study_best * 100:.2f}% (Trial {study_best_num:02d})".ljust(59) + "│")
            logging.info("└──────────────────────────────────────────────────────────┘")

            if self.current_optuna_save_dir:
                trial_dir = os.path.join(self.current_optuna_save_dir, "trials", f"trial_{trial.number:02d}")
                os.makedirs(trial_dir, exist_ok=True)

                params_dict = {
                    "C_penalty": c_penalty,
                    "scaling_factor": scaling_factor
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

        save_log_file = os.path.join(optuna_save_dir, "optuna_helm_hyperparam_search_best_results.txt")

        logging.info("==================================================")
        logging.info("         OPTUNA HELM CONFIGURATION SUMMARY        ")
        logging.info("==================================================")
        logging.info(f"  Target Dataset:       {self.cfg.get('dataset_name')}")
        logging.info(f"  Task ID:              {self.task_id}")
        logging.info(f"  Total Trials:         {n_trials}")
        logging.info(f"  C_penalty Range:      [{self.c_min:.6e} -> {self.c_max:.6e}]")
        logging.info(f"  Scaling Factor Range: [{self.scaling_min:.6f} -> {self.scaling_max:.6f}]")
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
            f.write("=== OPTUNA HELM HYPERPARAMETER SEARCH RESULTS ===\n")
            f.write(f"Dataset: {self.cfg.get('dataset_name')}\n")
            f.write(f"Best Accuracy: {best_value:.4f}\n")
            f.write("Best Parameters:\n")
            for key, value in best_params.items():
                f.write(f"  {key}: {value}\n")

        try:
            df = study.trials_dataframe()
            csv_path = os.path.join(optuna_save_dir, "optuna_helm_trials_history.csv")
            df.to_csv(csv_path, index=False)

            json_path = os.path.join(optuna_save_dir, "optuna_helm_best_params.json")
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

        save_log_file = os.path.join(ray_save_dir, "ray_helm_hyperparam_search_best_results.txt")

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

        h_cfg = self.cfg.get("hyperparamtuning", {}) if isinstance(self.cfg.get("hyperparamtuning"), dict) else {}

        scheduler = ASHAScheduler(
            metric=h_cfg.get("metric", "accuracy"),
            mode=h_cfg.get("mode", "max"),
            max_t=h_cfg.get("max_t", 100),
            grace_period=h_cfg.get("grace_period", 10),
            reduction_factor=h_cfg.get("reduction_factor", 2)
        )

        reporter = tune.CLIReporter(
            parameter_columns=["C_penalty", "scaling_factor"],
            metric_columns=["accuracy", "training_iteration"]
        )

        stopper = RedisAbortStopper(self.task_id)
        progress_callback = RayCeleryProgressCallback(self.celery_task, n_trials)

        try:
            result = tune.run(
                self.fit,
                stop=stopper,
                resources_per_trial={"cpu": 2, "gpu": 0},
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
                if k not in ["dataset_name", "task_id"]
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
            if ray.is_initialized():
                ray.shutdown()

    def tune_params(self, backend: str = "optuna", n_trials: int = 25) -> dict:
        """
        Performs hyperparameter tuning using specified framework (Optuna or Ray Tune).

        Args:
            backend: The search framework to use ('optuna' or 'ray').
            n_trials: Total number of search trials.

        Returns:
            dict: Dictionary with best_accuracy and best_params.
        """

        if backend.lower() == "optuna":
            return self.tune_params_optuna(n_trials=n_trials)
        elif backend.lower() == "ray":
            return self.tune_params_ray(n_trials=n_trials)
        else:
            raise ValueError(f"Unsupported backend '{backend}'. Choose 'optuna' or 'ray'.")


if __name__ == "__main__":
    try:
        hyper_par_tune = HyperparameterSearchHELM()
        hyper_par_tune.tune_params(backend="optuna", n_trials=25)
    except KeyboardInterrupt as kie:
        print(kie)