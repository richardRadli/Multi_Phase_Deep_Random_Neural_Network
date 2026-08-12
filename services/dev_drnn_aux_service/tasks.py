import os
import json
import logging
import sys
import numpy as np
from celery import Celery

from config.json_config import json_config_selector
from utils.utils import load_config_json

CELERY_BROKER = os.getenv("CELERY_BROKER_URL", "redis://redis_broker:6379/4")
CELERY_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis_broker:6379/4")

celery_app = Celery("ipmpdrnn_tasks", broker=CELERY_BROKER, backend=CELERY_BACKEND)

PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.data_paths import JSON_FILES_PATHS
from nn.mpdrnn_pruning.ipmpdrnn_multi_aux import IPMPDRNN
from nn.mpdrnn_pruning.param_search_ipmpdrnn import ParamSearchIPMPDRNN


def exponential_neurons(num_of_layers, num_of_neurons, decay_rate=0.5):
    if num_of_layers <= 0:
        raise ValueError("Number of layers must be greater than zero.")
    if num_of_neurons <= 0:
        raise ValueError("Number of neurons must be greater than zero.")

    layers = np.arange(num_of_layers)
    neuron_distribution = np.exp(-decay_rate * layers)
    neuron_distribution /= neuron_distribution.sum()
    neuron_distribution *= num_of_neurons
    neuron_distribution = np.round(neuron_distribution).astype(int)

    while neuron_distribution.sum() < num_of_neurons:
        for i in range(len(neuron_distribution)):
            if neuron_distribution[i] > 0:
                neuron_distribution[i] += 1
                if neuron_distribution.sum() >= num_of_neurons:
                    break
    return neuron_distribution.tolist()


@celery_app.task(bind=True, name="tasks.ipmpdrnn_task")
def ipmpdrnn_task(self, config: dict):
    logging.info("Starting IPMPDRNN task")
    task_id = self.request.id
    try:
        dataset_name = config["dataset_name"]
        method = config["method"]

        try:
            ipmpdrnn_config_path = JSON_FILES_PATHS.get_data_path("config_ipmpdrnn")
            with open(ipmpdrnn_config_path, "r") as f:
                raw_json = json.load(f)
        except Exception:
            raw_json = {}

        raw_json["dataset_name"] = dataset_name
        raw_json["activation"] = config["activation"]
        raw_json["number_of_tests"] = config["number_of_tests"]
        raw_json["seed"] = config["seed"]
        raw_json["method"] = method
        raw_json["subset_percentage"] = config.get("subset_percentage", 0.1)
        raw_json["num_aux_net"] = config.get("num_aux_net", 1)

        if method == "BASE":
            try:
                default_cfg = load_config_json(
                    json_schema_filename=json_config_selector("ipmpdrnn").get("schema"),
                    json_filename=json_config_selector("ipmpdrnn").get("config")
                )
                raw_json["mu"] = default_cfg.get("mu", 0)
                raw_json["sigma"] = default_cfg.get("sigma", 0.1)
            except Exception:
                raw_json["mu"] = 0
                raw_json["sigma"] = 0.1
        else:
            raw_json["mu"] = 0
            raw_json["sigma"] = config.get("sigma", 0.1)

        hidden_neurons = config.get("hidden_neurons")

        if hidden_neurons and isinstance(hidden_neurons, list):
            computed_exp_neurons = hidden_neurons
        else:
            computed_exp_neurons = exponential_neurons(
                num_of_layers=config.get("num_of_layers", 3),
                num_of_neurons=config.get("num_of_neurons", 100),
                decay_rate=config.get("decay_rate", 0.5)
            )

        simple_config = {k: v for k, v in raw_json.items() if not isinstance(v, dict)}
        nested_config = {k: v for k, v in raw_json.items() if isinstance(v, dict)}

        flattened_config = {}
        for key, value in nested_config.items():
            if key != 'hyperparamtuning' and dataset_name in value:
                flattened_config[key] = value[dataset_name]

        override_cfg = {**simple_config, **flattened_config}

        override_cfg["exp_neurons"] = computed_exp_neurons

        if config.get("penalty") is not None and method == "EXP_ORT_C":
            override_cfg["penalty"] = config["penalty"]

        if config.get("rcond") is not None:
            if "rcond" not in override_cfg or not isinstance(override_cfg["rcond"], dict):
                override_cfg["rcond"] = {}
            override_cfg["rcond"][method] = config["rcond"]

        evaluator = IPMPDRNN(override_cfg=override_cfg, celery_task=self)
        evaluator.main()

        is_aborted = self.backend.client.get(f"ipmpdrnn:abort:{task_id}")
        if is_aborted:
            self.update_state(state="REVOKED")
            return {"status": "ABORTED", "dataset_name": dataset_name}

        avg_metrics = getattr(evaluator, "averaged_metrics", [0.0] * 9)

        def get_metric_or_default(index, default=0.0):
            try:
                return float(avg_metrics[index])
            except (IndexError, TypeError, ValueError):
                return default

        return {
            "status": "SUCCESS",
            "dataset_name": dataset_name,
            "method": method,
            "output_file": getattr(evaluator, "save_filename", None),
            "metrics": {
                "train_accuracy": get_metric_or_default(0),
                "test_accuracy": get_metric_or_default(1),
                "train_precision": get_metric_or_default(2),
                "test_precision": get_metric_or_default(3),
                "train_recall": get_metric_or_default(4),
                "test_recall": get_metric_or_default(5),
                "train_f1_score": get_metric_or_default(6),
                "test_f1_score": get_metric_or_default(7),
                "training_time": get_metric_or_default(8)
            }
        }
    except Exception as e:
        logging.exception(f"IPMPDRNN error: {str(e)}")
        raise RuntimeError(str(e))


@celery_app.task(bind=True, name="tasks.ipmpdrnn_tune_task")
def ipmpdrnn_tune_task(self, config: dict):
    logging.info("Starting IPMPDRNN hyperparameter search task")
    task_id = self.request.id

    self.update_state(state="PROGRESS", meta={'status': "Exploring hyperparameter space...", "progress_percent": 0})
    try:
        dataset_name = config["dataset_name"]
        backend = config.get("backend", "optuna")
        n_trials = config.get("n_trials", 25)

        override_cfg = {
            "dataset_name": dataset_name,
            "task_id": task_id,
            "n_trials": n_trials,
            "seed": config.get("seed", False),
            "method": config.get("method", "BASE"),
            "activation": config.get("activation", "LeakyReLU"),
            "mu": config.get("mu", 0.0),
            "sigma": config.get("sigma", 0.1),
            "rcond_min": config.get("rcond_min", 1e-30),
            "rcond_max": config.get("rcond_max", 1e-1),
            "penalty_min": config.get("penalty_min", 0.1),
            "penalty_max": config.get("penalty_max", 30.0),
            "sp_min": config.get("sp_min", 0.1),
            "sp_max": config.get("sp_max", 0.9),
            "num_aux_min": config.get("num_aux_min", 1),
            "num_aux_max": config.get("num_aux_max", 5),
            "l1_min": config.get("l1_min", 600),
            "l1_max": config.get("l1_max", 1000),
            "l2_min": config.get("l2_min", 200),
            "l2_max": config.get("l2_max", 500),
            "l3_min": config.get("l3_min", 50),
            "l3_max": config.get("l3_max", 100)
        }

        searcher = ParamSearchIPMPDRNN(override_cfg=override_cfg, celery_task=self)
        res = searcher.tune_params(backend=backend, n_trials=n_trials)

        is_aborted = self.backend.client.get(f"ipmpdrnn:abort:{task_id}")
        if is_aborted:
            self.update_state(state="REVOKED")
            return {"status": "ABORTED", "dataset_name": dataset_name, "backend": backend}

        return {
            "status": "SUCCESS",
            "dataset_name": dataset_name,
            "backend": res.get("backend", backend),
            "best_accuracy": res.get("best_accuracy", 0.0),
            "best_params": res.get("best_params", {})
        }
    except Exception as e:
        if "ABORTED" in str(e):
            self.update_state(state="REVOKED")
            return {"status": "ABORTED", "dataset_name": config.get("dataset_name")}
        logging.exception(f"IPMPDRNN hyperparameter search error: {str(e)}")
        raise RuntimeError(str(e))