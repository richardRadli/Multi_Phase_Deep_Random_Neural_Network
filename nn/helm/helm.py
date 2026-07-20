import logging
import os

import numpy as np
from tqdm import tqdm

from config.dataset_config import helm_paths_config
from nn.helm.base_class_helm import HELMBase
from utils.utils import (
    average_columns_in_excel,
    create_timestamp,
    insert_data_to_excel,
    reorder_metrics_lists,
    plot_confusion_matrix_helm
)


class HELM(HELMBase):
    def __init__(self, override_cfg: dict = None, celery_task = None):
        """
        Initializes the HELM class and sets up file paths and hyperparameter configurations.

        Returns:
            None
        """
        super().__init__(override_cfg=override_cfg, celery_task=celery_task)

        self.timestamp = create_timestamp()
        self.dataset_name = self.cfg.get("dataset_name")
        self.method = self.cfg.get("method", "BASE")
        self.helm_config = helm_paths_config(self.dataset_name)

        base_results_dir = self.helm_config.get("path_to_results")

        self.excel_dir = os.path.join(base_results_dir, "excel")
        self.cm_dir = os.path.join(base_results_dir, "confusion_matrix")

        self.filename = os.path.join(
            self.excel_dir,
            f"{self.timestamp}_{self.dataset_name}_dataset.xlsx"
        )

        self.hyperparam_config = {
            "C_penalty": self.cfg.get("penalty"),
            "scaling_factor": self.cfg.get("scaling_factor"),
        }

    def main(self) -> None:
        """
        Runs the main evaluation loop for the HELM.

        Iterates through the number of test cycles, performs training and evaluation,
        and saves the results to an Excel file.

        Returns:
            None
        """
        os.makedirs(self.excel_dir, exist_ok=True)
        os.makedirs(self.cm_dir, exist_ok=True)

        num_tests = self.cfg.get("num_tests", 1)
        all_cycle_metrics = []

        for idx in tqdm(range(num_tests), desc="Evaluation"):
            if self.celery_task:
                is_aborted = self.celery_task.backend.client.get(f"helm:abort:{self.celery_task.request.id}")
                if is_aborted:
                    logging.info("HELM testing series abort signal detected mid-cycle. Breaking loop.")
                    break

            progress_percent = int((idx / num_tests) * 100)
            self.celery_task.update_state(
                state="PROGRESS",
                meta={
                    "status": f"Evaluating hierarchy: test series {idx + 1}/{num_tests}...",
                    "telemetry": {
                        "current_cycle": idx + 1,
                        "total_cycles": num_tests,
                        "progress_percent": progress_percent
                    }
                }
            )

            t3, beta, beta1, beta2, l3, ps1, ps2 = self.train(config=self.hyperparam_config)
            training_metrics = self.training_accuracy(t3, beta)
            testing_metrics = self.evaluation(beta, beta1, beta2, l3, ps1, ps2, self.test_loader)
            train_cm_list = getattr(self, "train_cm_list", [])
            test_cm_list = getattr(self, "test_cm_list", [])

            prefix = f"{self.timestamp}_cycle_{idx}_"

            if train_cm_list:
                plot_confusion_matrix_helm(
                    cm_list=train_cm_list,
                    path_to_plot=self.cm_dir,
                    name_of_dataset=self.dataset_name,
                    operation="train",
                    prefix=prefix,
                    labels=self.labels
                )

            if test_cm_list:
                plot_confusion_matrix_helm(
                    cm_list=test_cm_list,
                    path_to_plot=self.cm_dir,
                    name_of_dataset=self.dataset_name,
                    operation="test",
                    prefix=prefix,
                    labels=self.labels
                )

            metrics = reorder_metrics_lists(train_metrics=training_metrics,
                                            test_metrics=testing_metrics)
            insert_data_to_excel(self.filename, self.dataset_name, idx + 2, metrics)

            all_cycle_metrics.append(metrics[0])

        if all_cycle_metrics:
            self.averaged_metrics = np.mean(all_cycle_metrics, axis=0).tolist()

        if os.path.exists(self.filename):
            average_columns_in_excel(self.filename)


if __name__ == "__main__":
    helm = HELM()
    helm.main()