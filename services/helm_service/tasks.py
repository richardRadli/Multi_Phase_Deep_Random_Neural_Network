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
    logging.info("Starting unified HELM task")
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

        if config.get("penalty") is not None:
            override_cfg["penalty"] = config["penalty"]
            logging.info(f"Overriding gyári penalty value to: {config['penalty']}")


        if config.get("scaling_factor") is not None:
            override_cfg["scaling_factor"] = config["scaling_factor"]
            logging.info(f"Overriding gyári scaling_factor value to: {config['scaling_factor']}")

        evaluator = HELM(override_cfg=override_cfg, celery_task=self)
        evaluator.main()

        is_aborted = self.backend.client.get(f"helm:abort:{task_id}")
        if is_aborted:
            self.update_state(state="REVOKED")
            return {"status": "ABORTED", "mode": "standard", "dataset_name": dataset_name}

        return {
            "status": "SUCCESS",
            "mode": "series" if num_tests > 1 else "single",
            "dataset_name": dataset_name,
            "output_file": getattr(evaluator, "filename", None)
        }
    except Exception as e:
        logging.exception(f"HELM task encountered an error: {str(e)}")
        raise RuntimeError(f"HELM execution failed: {str(e)}")