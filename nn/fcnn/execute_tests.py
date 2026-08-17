import logging

import colorama
import os
import numpy as np
from tqdm import tqdm

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import  fcnn_paths_configs
from nn.fcnn.eval_fcnn import EvalFCNN
from nn.fcnn.train_fcnn import TrainFCNN
from utils.utils import create_timestamp, insert_data_to_excel, load_config_json, average_columns_in_excel
from typing import Tuple, List

def main(override_cfg: dict = None, celery_task = None) -> Tuple[str, List[float]]:
    """
    Main function to load configuration, run training and evaluation cycles, and save results to an Excel file.

    Returns:
        None: The function does not return any value but performs training, evaluation, and data logging.
    """

    timestamp = create_timestamp()

    if override_cfg is not None:
        cfg = override_cfg
    else:
        cfg = (
            load_config_json(json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_fcnn"),
                             json_filename=JSON_FILES_PATHS.get_data_path("config_fcnn"))
        )

    dataset_name = cfg.get("dataset_name")
    batch_size = cfg.get("batch_size")
    hidden_neurons = cfg.get("hidden_neurons")
    device = cfg.get("device")
    optimizer = cfg.get("optimizer")
    optimization = cfg.get("optimization")

    if cfg.get("learning_rate") is not None:
        lr = cfg.get("learning_rate")
    else:
        lr = optimization.get(optimizer).get("learning_rate").get(dataset_name)

    num_tests = cfg.get("num_tests", 1)

    fcnn_config = fcnn_paths_configs(dataset_name)
    base_results_dir = fcnn_config.get("saved_results")

    excel_dir = os.path.join(base_results_dir, "excel")
    os.makedirs(excel_dir, exist_ok=True)

    filename = os.path.join(excel_dir, f"{timestamp}_bs_{batch_size}_hn_{hidden_neurons}_lr_{lr}_device_{device}.xlsx")

    collected_data = []
    all_series_metrics = []

    for i in tqdm(range(cfg.get("num_tests")), desc="Testing cycle"):
        if celery_task:
            is_aborted = celery_task.backend.client.get(f"fcnn:abort:{celery_task.request.id}")
            if is_aborted:
                logging.info("FCNN testing series abort signal detected mid-cycle. Breaking loop.")
                break

        def on_epoch_end(current_epoch: int, total_epochs: int):
            if celery_task:
                cycle_progress = current_epoch / total_epochs
                overall_pct = round(((i + cycle_progress) / num_tests) * 100, 1)
                celery_task.update_state(
                    state="PROGRESS",
                    meta={
                        "status": f"Testing cycle {i + 1}/{num_tests} - Epoch {current_epoch}/{total_epochs}",
                        "current_epoch": current_epoch,
                        "total_epochs": total_epochs,
                        "current_cycle": i + 1,
                        "total_cycles": num_tests,
                        "progress_percent": overall_pct
                    }
                )

        train_fcnn = TrainFCNN(override_cfg=cfg, celery_task=celery_task)
        train_fcnn.epoch_callback = on_epoch_end
        train_fcnn.fit()
        training_time = train_fcnn.fit.execution_time

        if celery_task:
            is_aborted = celery_task.backend.client.get(f"fcnn:abort:{celery_task.request.id}")
            if is_aborted:
                break

        eval_fcnn = EvalFCNN(override_cfg=cfg)
        eval_fcnn.main()

        cycle_metric_tuple = (
            getattr(eval_fcnn, "train_accuracy", 0.0),
            getattr(eval_fcnn, "test_accuracy", 0.0),
            getattr(eval_fcnn, "train_precision", 0.0),
            getattr(eval_fcnn, "test_precision", 0.0),
            getattr(eval_fcnn, "train_recall", 0.0),
            getattr(eval_fcnn, "test_recall", 0.0),
            getattr(eval_fcnn, "train_f1sore", 0.0),
            getattr(eval_fcnn, "test_f1sore", 0.0),
            training_time
        )

        collected_data.append(cycle_metric_tuple)
        all_series_metrics.append(cycle_metric_tuple)

        insert_data_to_excel(filename=filename,
                             dataset_name=dataset_name,
                             row=i + 2,
                             data=collected_data)

        collected_data.clear()

    if os.path.exists(filename):
        average_columns_in_excel(filename)

    if all_series_metrics:
        avg_metrics = np.mean(all_series_metrics, axis=0).tolist()
    else:
        avg_metrics = [0.0] * 9

    return filename, avg_metrics


if __name__ == '__main__':
    main()
