import os
import json
import time
import logging
import sys
from celery import Celery
from celery.exceptions import Ignore
from nn.fcnn.execute_tests import main as run_execute_tests

CELERY_BROKER = os.getenv("CELERY_BROKER_URL", "redis://redis_broker:6379/1")
CELERY_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis_broker:6379/1")

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
from nn.fcnn.param_search_fcnn import HyperparameterSearch


@celery_app.task(bind=True, name="tasks.train_fcnn_task")
def train_fcnn_task(self, config: dict):
    logging.info("Starting FCNN training task")

    task_id = self.request.id
    dataset_name = config["dataset_name"]
    seed = config["seed"]
    epochs = config["epochs"]

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
        if "patience" in config:
            raw_json["patience"] = config["patience"]
        if "batch_size" in config:
            raw_json["batch_size"] = config["batch_size"]
        if "optimizer" in config:
            raw_json["optimizer"] = config["optimizer"]

        simple_config = {k: v for k, v in raw_json.items() if not isinstance(v, dict)}

        hidden_neurons = config.get("hidden_neurons") or raw_json.get("hidden_neurons", {}).get(dataset_name, 500)
        batch_size = config.get("batch_size") or raw_json.get("batch_size", {}).get(dataset_name, 64)

        override_cfg = {
            **simple_config,
            "hidden_neurons": hidden_neurons,
            "batch_size": batch_size,
            "learning_rate": config.get("learning_rate"),
            "momentum": config.get("momentum"),
            "optimization": raw_json.get("optimization")
        }

        trainer = TrainFCNN(override_cfg=override_cfg, celery_task=self)
        trainer.epoch_callback = on_epoch_end

        from config.dataset_config import fcnn_paths_configs
        fcnn_ds_cfg = fcnn_paths_configs(dataset_name)

        meta_dir = os.path.join(fcnn_ds_cfg.get("saved_results"), "metadata")
        os.makedirs(meta_dir, exist_ok=True)

        params_path = os.path.join(meta_dir, f"run_{task_id}.json")

        config_data = {
            "task_id": task_id,
            "dataset_name": dataset_name,
            "seed": seed,
            "epochs": epochs,
            "patience": raw_json.get("patience", 10),
            "status": "running",
            "start_time": time.time(),
            "start_time_str": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)

        trainer.fit()

        execution_time = trainer.fit.execution_time
        is_aborted = self.backend.client.get(f"fcnn:abort:{task_id}")

        if is_aborted:
            config_data["status"] = "aborted"
            config_data["final_epoch"] = getattr(trainer, "current_epoch_run", 0)
            config_data["end_time_str"] = time.strftime("%Y-%m-%d %H:%M:%S")
            config_data["execution_time_seconds"] = round(execution_time, 4)

            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=4)

            abort_payload = {
                "status": "ABORTED",
                "execution_time_seconds": round(execution_time, 4),
                "total_epochs_run": getattr(trainer, "current_epoch_run", 0),
                "message": f"Training gracefully stopped at epoch {getattr(trainer, 'current_epoch_run', 0)}."
            }
            self.update_state(state="ABORTED", meta=abort_payload)
            raise Ignore()

        else:
            config_data["status"] = "completed"
            config_data["final_epoch"] = epochs
            config_data["end_time_str"] = time.strftime("%Y-%m-%d %H:%M:%S")
            config_data["execution_time_seconds"] = round(execution_time, 4)

            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=4)

        return {
            "status": "SUCCESS",
            "execution_time_seconds": round(execution_time, 4),
            "total_epochs_run": getattr(trainer, "current_epoch_run", 0)
        }

    except Exception as e:
        if isinstance(e, Ignore):
            raise e
        logging.exception(f"FCNN training task encountered an error: {str(e)}")
        try:
            target_json = params_path if params_path else os.path.join(fcnn_ds_cfg.get("saved_results"), "metadata",
                                                                       f"run_failed_{task_id}.json")
            config_data["status"] = "failed"
            config_data["error"] = str(e)
            config_data["end_time_str"] = time.strftime("%Y-%m-%d %H:%M:%S")
            config_data["execution_time_seconds"] = round(time.time() - config_data.get("start_time", time.time()), 4)
            config_data["final_epoch"] = getattr(trainer, "current_epoch_run", 0) if trainer else 0
            with open(target_json, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=4)
        except Exception:
            pass
        raise e
    finally:
        self.backend.client.srem("fcnn:active_tasks", task_id)


@celery_app.task(bind=True, name="tasks.test_fcnn_task")
def test_fcnn_task(self, config: dict):
    logging.info("Starting FCNN evaluation task (Strictly Pure Testing Mode)")
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
        raw_json["seed"] = seed
        raw_json["epochs"] = epochs
        if "patience" in config:
            raw_json["patience"] = config["patience"]
        if "batch_size" in config:
            raw_json["batch_size"] = config["batch_size"]
        if "optimizer" in config:
            raw_json["optimizer"] = config["optimizer"]

        simple_config = {k: v for k, v in raw_json.items() if not isinstance(v, dict)}

        hidden_neurons = config.get("hidden_neurons") or raw_json.get("hidden_neurons", {}).get(dataset_name, 500)
        batch_size = config.get("batch_size") or raw_json.get("batch_size", {}).get(dataset_name, 64)

        override_cfg = {
            **simple_config,
            "hidden_neurons": hidden_neurons,
            "batch_size": batch_size,
            "learning_rate": config.get("learning_rate"),
            "momentum": config.get("momentum"),
            "model_checkpoint": config.get("model_checkpoint"),
            "optimization": raw_json.get("optimization")
        }

        def safe_float(val):
            return float(val) if val is not None else 0.0

        if not series_mode:
            evaluator = EvalFCNN(override_cfg=override_cfg)
            evaluator.main()

            return {
                "status": "SUCCESS",
                "mode": "single",
                "dataset_name": dataset_name,
                "metrics": {
                    "accuracy": safe_float(getattr(evaluator, "test_accuracy", None)),
                    "precision": safe_float(getattr(evaluator, "test_precision", None)),
                    "recall": safe_float(getattr(evaluator, "test_recall", None)),
                    "f1_score": safe_float(getattr(evaluator, "test_f1sore", None)),
                    "confusion_matrix": getattr(evaluator, "test_cm", None).tolist() if getattr(evaluator, "test_cm",
                                                                                                None) is not None else None
                }
            }
        else:
            filename, avg_metrics = run_execute_tests(override_cfg=override_cfg, celery_task=self)

            is_aborted = self.backend.client.get(f"fcnn:abort:{task_id}")
            if is_aborted:
                self.update_state(
                    state="ABORTED",
                    meta={
                        "status": "ABORTED",
                        "mode": "series",
                        "dataset_name": dataset_name,
                        "message": "FCNN test series was manually aborted."
                    }
                )
                raise Ignore()

            return {
                "status": "SUCCESS",
                "mode": "series",
                "dataset_name": dataset_name,
                "output_file": filename,
                "metrics": {
                    "train_accuracy": float(avg_metrics[0]),
                    "test_accuracy": float(avg_metrics[1]),
                    "train_precision": float(avg_metrics[2]),
                    "test_precision": float(avg_metrics[3]),
                    "train_recall": float(avg_metrics[4]),
                    "test_recall": float(avg_metrics[5]),
                    "train_f1_score": float(avg_metrics[6]),
                    "test_f1_score": float(avg_metrics[7]),
                    "training_time": float(avg_metrics[8]),
                    "accuracy": float(avg_metrics[1]),
                    "precision": float(avg_metrics[3]),
                    "recall": float(avg_metrics[5]),
                    "f1_score": float(avg_metrics[7])
                },
                "message": f"Excel test series report generated with {num_tests} cycles."
            }

    except Exception as e:
        logging.exception(f"FCNN evaluation task encountered an error: {str(e)}")
        raise RuntimeError(f"FCNN evaluation failed: {str(e)}")


@celery_app.task(bind=True, name="tasks.tune_fcnn_task")
def tune_fcnn_task(self, config: dict):
    logging.info("Starting FCNN hyperparameter search task")
    self.update_state(state="PROGRESS", meta={"status": "Initializing hyperparameter search...", "progress_percent": 0})

    try:
        config["task_id"] = self.request.id

        backend = config.get("backend", "optuna")
        n_trials = config.get("n_trials", 25)

        searcher = HyperparameterSearch(override_cfg=config, celery_task=self)
        results = searcher.tune_params(backend=backend, n_trials=n_trials)

        return {
            "status": "SUCCESS",
            "dataset_name": config["dataset_name"],
            "backend": backend,
            "best_accuracy": results["best_accuracy"],
            "best_params": results["best_params"]
        }
    except Exception as e:
        if "ABORTED" in str(e):
            logging.info("Hyperparameter tuning gracefully aborted by user.")
            self.update_state(state="ABORTED", meta={"message": "Tuning process aborted."})
            raise Ignore()
        logging.exception(f"FCNN hyperparameter tuning task failed: {str(e)}")
        raise RuntimeError(f"Tuning failed: {str(e)}")