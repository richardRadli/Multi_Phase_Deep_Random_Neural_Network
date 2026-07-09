import os
import json
import time
import logging
import sys
from celery import Celery

CELERY_BROKER = os.getenv("CELERY_BROKER_URL", "redis://redis_broker:6379/0")
CELERY_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis_broker:6379/0")

celery_app = Celery("fcnn_tasks", broker=CELERY_BROKER, backend=CELERY_BACKEND)

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
from nn.fcnn.train_fcnn import TrainFCNN
from nn.fcnn.eval_fcnn import EvalFCNN

@celery_app.task(bind=True, name="task.train_fcnn_task")
def train_fcnn_task(self, config: dict):
    logging.info("Starting FCNN training task")

    task_id = self.request.id
    dataset_name = config["dataset_name"]
    seed = config["seed"]
    epochs = config["epochs"]

    STORAGE_ROOT = os.getenv("STORAGE_ROOT", "/app/storage")
    params_path = os.path.join(STORAGE_ROOT, f"training_params_{task_id}.json")

    config_data = {
        "task_id": task_id,
        "dataset_name": dataset_name,
        "seed": seed,
        "epochs": epochs,
        "status": "running",
        "start_time": time.time(),
        "start_time_str": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(params_path, "w") as f:
        json.dump(config_data, f, indent=4)

    def on_epoch_end(current_epoch: int, total_epochs: int):
        self.update_state(
            state="PROGRESS",
            meta={
                "current_epoch": current_epoch,
                "total_epochs": total_epochs,
                "progress_percent": round((current_epoch / total_epochs) * 100, 1)
            }
        )

    trainer = None
    try:
        fcnn_config_path = JSON_FILES_PATHS.get_data_path("config_fcnn")
        with open(fcnn_config_path, "r") as f:
            raw_json = json.load(f)

        raw_json["dataset_name"] = dataset_name
        raw_json["seed"] = seed
        raw_json["epochs"] = epochs

        simple_config = {k: v for k, v in raw_json.items() if not isinstance(v, dict)}
        nested_config = {k: v for k, v in raw_json.items() if isinstance(v, dict)}

        flattened_config = {}
        for key, value in nested_config.items():
            if key != 'hyperparamtuning' and dataset_name in value:
                flattened_config[key] = value[dataset_name]

        override_cfg = {**simple_config, **flattened_config}

        trainer = TrainFCNN(override_cfg=override_cfg, celery_task=self)
        trainer.epoch_callback = on_epoch_end
        trainer.fit()

        execution_time = trainer.fit.execution_time

        is_aborted = self.backend.client.get(f"fcnn:abort:{task_id}")

        if is_aborted:
            config_data["status"] = "aborted"
            config_data["final_epoch"] = getattr(trainer, "current_epoch_run", 0)
            self.update_state(state="REVOKED")
        else:
            config_data["status"] = "completed"
            config_data["final_epoch"] = epochs

        config_data["end_time_str"] = time.strftime("%Y-%m-%d %H:%M:%S")
        config_data["execution_time_seconds"] = round(execution_time, 4)

        with open(params_path, "w") as f:
            json.dump(config_data, f, indent=4)

        return {
            "status": "SUCCESS" if not is_aborted else "ABORTED",
            "execution_time_seconds": round(execution_time, 4),
            "total_epochs_run": getattr(trainer, "current_epoch_run", 0)
        }

    except Exception as e:
        logging.exception(f"FCNN training task encountered an error: {str(e)}")
        config_data["status"] = "failed"
        config_data["error"] = str(e)
        config_data["end_time_str"] = time.strftime("%Y-%m-%d %H:%M:%S")
        config_data["execution_time_seconds"] = round(time.time() - config_data["start_time"], 4)
        config_data["final_epoch"] = getattr(trainer, "current_epoch_run", 0)

        with open(params_path, "w") as f:
            json.dump(config_data, f, indent=4)
        raise e
    finally:
        self.backend.client.srem("fcnn:active_tasks", task_id)


@celery_app.task(bind=True, name="tasks.test_fcnn_task")
def test_fcnn_task(self, config: dict):
    logging.info("Starting FCNN evaluation task")
    self.update_state(state="PROGRESS", meta={'status': "Initializing weights and dataset for evaluation"})
    task_id = self.request.id
    try:
        dataset_name = config["dataset_name"]
        batch_size = config["batch_size"]
        seed = config["seed"]
        series_mode = config["series_mode"]
        num_tests = config.get("num_tests", 1)
        epochs = config.get("epochs", 1000)

        fcnn_config_path = JSON_FILES_PATHS.get_data_path("config_fcnn")
        with open(fcnn_config_path, "r") as f:
            raw_json = json.load(f)

        raw_json["dataset_name"] = dataset_name
        raw_json["batch_size"] = batch_size
        raw_json["seed"] = seed
        raw_json["num_tests"] = num_tests
        raw_json["epochs"] = epochs

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
                "status": "SUCCESS",
                "mode": "single",
                "dataset_name": dataset_name,
                "metrics": {
                    "accuracy": getattr(evaluator, "test_accuracy", None),
                    "precision": getattr(evaluator, "test_precision", None),
                    "recall": getattr(evaluator, "test_recall", None),
                    "f1_score": getattr(evaluator, "test_f1sore", None),
                    "confusion_matrix": getattr(evaluator, "test_cm", None).tolist() if getattr(evaluator, "test_cm",
                                                                                                None) is not None else None
                }
            }
        else:
            from nn.fcnn.execute_tests import main as run_execute_tests
            run_execute_tests(override_cfg=override_cfg, celery_task=self)

            is_aborted = self.backend.client.get(f"fcnn:abort:{task_id}")
            if is_aborted:
                self.update_state(state="REVOKED")
                return {
                    "status": "ABORTED",
                    "mode": "series",
                    "dataset_name": dataset_name,
                    "message": "FCNN test series was manually aborted."
                }

            return {
                "status": "SUCCESS",
                "mode": "series",
                "dataset_name": dataset_name,
                "message": f"Excel test series report generated with {num_tests} cycles and columns averaged successfully."
            }
    except Exception as e:
        logging.exception(f"FCNN evaluation task encountered an error: {str(e)}")
        raise RuntimeError(f"FCNN evaluation failed: {str(e)}")

