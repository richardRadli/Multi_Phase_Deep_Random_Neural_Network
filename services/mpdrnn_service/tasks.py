import os
import json
import logging
import sys
from celery import Celery

CELERY_BROKER = os.getenv("CELERY_BROKER_URL", "redis://redis_broker:6379/0")
CELERY_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis_broker:6379/0")

celery_app = Celery("mpdrnn_tasks", broker=CELERY_BROKER, backend=CELERY_BACKEND)

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
from nn.mpdrnn.mpdrnn import MPDRNN

@celery_app.task(bind=True, name="tasks.mpdrnn_task")
def mpdrnn_task(self, config: dict):
    logging.info("Starting unified MPDRNN task")
    task_id = self.request.id
    self.update_state(state="PROGRESS", meta={'status': "Running Multi-Phase Deep Randomized computations"})
    try:
        dataset_name = config["dataset_name"]
        method = config["method"]

        mpdrnn_config_path = JSON_FILES_PATHS.get_data_path("config_mpdrnn")
        with open(mpdrnn_config_path, "r") as f:
            raw_json = json.load(f)

        raw_json["dataset_name"] = dataset_name
        raw_json["activation"] = config["activation"]
        raw_json["number_of_tests"] = config["number_of_tests"]
        raw_json["seed"] = config["seed"]
        raw_json["method"] = method
        raw_json["mu"] = 0  # szigorúan fixálva 0-ra
        raw_json["sigma"] = config["sigma"]

        simple_config = {k: v for k, v in raw_json.items() if not isinstance(v, dict)}
        nested_config = {k: v for k, v in raw_json.items() if isinstance(v, dict)}

        flattened_config = {}
        for key, value in nested_config.items():
            if dataset_name in value:
                if key == "rcond":
                    flattened_config[key] = value[dataset_name]
                else:
                    flattened_config[key] = value[dataset_name]

        override_cfg = {**simple_config, **flattened_config}

        if config.get("penalty") is not None:
            override_cfg["penalty"] = config["penalty"]
        if config.get("eq_neurons") is not None:
            override_cfg["eq_neurons"] = config["eq_neurons"]
        if config.get("exp_neurons") is not None:
            override_cfg["exp_neurons"] = config["exp_neurons"]

        if config.get("rcond") is not None:
            override_cfg["rcond"][method] = config["rcond"]

        evaluator = MPDRNN(override_cfg=override_cfg, celery_task=self)
        evaluator.main()

        is_aborted = self.backend.client.get(f"mpdrnn:abort:{task_id}")
        if is_aborted:
            self.update_state(state="REVOKED")
            return {"status": "ABORTED", "mode": "standard", "dataset_name": dataset_name}

        return {
            "status": "SUCCESS",
            "dataset_name": dataset_name,
            "method": method,
            "output_file": getattr(evaluator, "filename", None)
        }
    except Exception as e:
        logging.exception(f"MPDRNN task encountered an error: {str(e)}")
        raise RuntimeError(f"MPDRNN execution failed: {str(e)}")