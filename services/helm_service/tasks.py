import os
import json
import logging
import sys
from celery import Celery

CELERY_BROKER = os.getenv("CELERY_BROKER_URL", "redis://redis_broker:6379/2")
CELERY_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis_broker:6379/2")

celery_app = Celery("helm_tasks", broker=CELERY_BROKER, backend=CELERY_BACKEND)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.data_paths import JSON_FILES_PATHS
from nn.helm.helm import HELM


@celery_app.task(bind=True, name="tasks.helm_task")
def helm_task(self, config: dict):
    logging.info("Starting updated HELM task based on common architecture specs")
    task_id = self.request.id

    self.update_state(state="PROGRESS", meta={'status': "Running analytical matrix computations"})
    try:
        dataset_name = config["dataset_name"]
        seed = config["seed"]
        num_tests = config.get("num_tests", 1)

        helm_config_path = JSON_FILES_PATHS.get_data_path("config_helm")
        with open(helm_config_path, "r") as f:
            raw_json = json.load(f)

        raw_json["dataset_name"] = dataset_name
        raw_json["seed"] = seed
        raw_json["num_tests"] = num_tests

        simple_config = {k: v for k, v in raw_json.items() if not isinstance(v, dict)}
        nested_config = {k: v for k, v in raw_json.items() if isinstance(v, dict)}

        flattened_config = {}
        for key, value in nested_config.items():
            if key != 'hyperparamtuning' and dataset_name in value:
                flattened_config[key] = value[dataset_name]

        override_cfg = {**simple_config, **flattened_config}

        if not override_cfg.get("hidden_neurons"):
            override_cfg["hidden_neurons"] = raw_json.get("hidden_neurons", {}).get(dataset_name, [100, 50, 25])

        if config.get("penalty") is not None:
            override_cfg["penalty"] = config["penalty"]
            logging.info(f"Overriding default penalty value to: {config['penalty']}")

        if config.get("scaling_factor") is not None:
            override_cfg["scaling_factor"] = config["scaling_factor"]
            logging.info(f"Overriding default scaling_factor value to: {config['scaling_factor']}")

        override_cfg["method"] = "HELM"

        evaluator = HELM(override_cfg=override_cfg, celery_task=self)
        evaluator.main()

        is_aborted = self.backend.client.get(f"helm:abort:{task_id}")
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
            "method": "HELM",
            "output_file": getattr(evaluator, "filename", None),
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
        logging.exception(f"HELM error: {str(e)}")
        raise RuntimeError(str(e))