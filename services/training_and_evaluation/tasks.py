import os
import json
import time
import logging
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("training_tasks", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_track_started=True,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
)

import sys

PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.data_paths import ConfigFilePaths
from nn.fcnn.train_fcnn import TrainFCNN
from nn.fcnn.eval_fcnn import EvalFCNN


@celery_app.task(bind=True, name="task.train_fcnn_task")
def train_fcnn_task(dataset_name: str, seed: bool, epochs: int, num_tests: int):
    logging.info(f"Celery task started for dataset_name={dataset_name}")

    config_data = {
        "dataset_name": dataset_name,
        "seed": seed,
        "epochs": epochs,
        "num_tests": num_tests,
        "status": "running",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    params_path = os.path.join(PROJECT_ROOT, "training_params.json")
    with open(params_path, "w") as f:
        json.dump(config_data, f, indent=4)

    try:
        fcnn_config_path = ConfigFilePaths().get_data_path("config_fcnn")
        with open(fcnn_config_path, "r") as f:
            raw_json = json.load(f)

        raw_json["dataset_name"] = dataset_name
        raw_json["patience"] = patience
        raw_json["seed"] = seed
        raw_json["epochs"] = epochs
        raw_json["num_tests"] = num_tests

        simple_config = {k: v for k, v in raw_json.items() if not isinstance(v, dict)}
        nested_config = {k: v for k, v in raw_json.items() if isinstance(v, dict)}

        flattened_config = {}
        for key, value in nested_config.items():
            if dataset_name in value:
                flattened_config[key] = value[dataset_name]

        override_cfg = {**simple_config, **flattened_config}

        trainer = TrainFCNN(override_cfg=override_cfg)
        trainer.fit()

        execution_time = trainer.fit.execution_time

        config_data["status"] = "completed"
        config_data["execution_time_seconds"] = round(execution_time, 4)
        config_data["stopped_at_epoch"] = getattr(trainer, "stopped_at_epoch", epochs)

        return {"status": "success", "execution_time_seconds": execution_time,
                "stopped_at_epoch": config_data["stopped_at_epoch"]}

    except Exception as e:
        config_data["status"] = "failed"
        config_data["error"] = str(e)
        raise e
    finally:
        config_data["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(params_path, "w") as f:
            json.dump(config_data, f, indent=4)


@celery_app.task(name="tasks.test_fcnn_task")
def test_fcnn_task(dataset_name: str, batch_size: int, seed: bool, series_mode: bool):
    logging.info(f"Celery task started: Evaluating FCNN. Series mode: {series_mode}")

    fcnn_config_path = ConfigFilePaths().get_data_path("config_fcnn")
    with open(fcnn_config_path, "r") as f:
        raw_json = json.load(f)

    raw_json["dataset_name"] = dataset_name
    if "batch_size" in raw_json and dataset_name in raw_json["batch_size"]:
        raw_json["batch_size"][dataset_name] = batch_size
    raw_json["seed"] = seed

    simple_config = {k: v for k, v in raw_json.items() if not isinstance(v, dict)}
    nested_config = {k: v for k, v in raw_json.items() if isinstance(v, dict)}

    flattened_config = {}
    for key, value in nested_config.items():
        if key != 'hyperparamtuning' and dataset_name in value:
            flattened_config[key] = value[dataset_name]

    override_cfg = {**simple_config, **flattened_config}

    if not series_mode:
        evaluator = EvalFCNN(override_cfg=override_cfg)
        evaluator.main()

        return {
            "mode": "single",
            "dataset_name": dataset_name,
            "metrics": {
                "accuracy": getattr(evaluator, "test_accuracy", None),
                "precision": getattr(evaluator, "test_precision", None),
                "recall": getattr(evaluator, "test_recall", None),
                "f1_score": getattr(evaluator, "test_f1sore", None)
            }
        }
    else:
        from nn.fcnn.execute_tests import main as run_execute_tests
        run_execute_tests()
        return {
            "mode": "series",
            "dataset_name": dataset_name,
            "status": "Excel test series report generated successfully."
        }