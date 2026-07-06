import os

from tqdm import tqdm

from config.dataset_config import helm_paths_config
from nn.helm.base_class_helm import HELMBase
from utils.utils import average_columns_in_excel, create_timestamp, insert_data_to_excel, reorder_metrics_lists


class HELM(HELMBase):
    def __init__(self):
        """
        Initializes the HELM class and sets up file paths and hyperparameter configurations.

        Returns:
            None
        """
        super().__init__()

        timestamp = create_timestamp()
        dataset_name = self.cfg.get("dataset_name")
        helm_config = helm_paths_config(dataset_name)

        self.filename = \
            os.path.join(helm_config.get("path_to_results"), f"{timestamp}_{dataset_name}_dataset.xlsx")

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

        for idx in tqdm(range(self.cfg.get("num_tests")), desc="Evaluation"):
            t3, beta, beta1, beta2, l3, ps1, ps2 = self.train(config=self.hyperparam_config)
            training_metrics = self.training_accuracy(t3, beta)
            testing_metrics = self.evaluation(beta, beta1, beta2, l3, ps1, ps2, self.test_loader)

            metrics = reorder_metrics_lists(train_metrics=training_metrics,
                                            test_metrics=testing_metrics)
            insert_data_to_excel(self.filename, self.cfg.get("dataset_name"), idx + 2, metrics)

        average_columns_in_excel(self.filename)


if __name__ == "__main__":
    helm = HELM()
    helm.main()