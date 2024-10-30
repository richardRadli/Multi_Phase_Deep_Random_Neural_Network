import colorama
import os
import logging

from tqdm import tqdm
from config.dataset_config import general_dataset_configs, drnn_paths_config
from nn.mpdrnn.base_class_mpdrnn import BaseMPDRNN
from nn.models.model_selector import ModelFactory
from utils.utils import (average_columns_in_excel, create_timestamp, get_num_of_neurons, insert_data_to_excel,
                         reorder_metrics_lists)


class MPDRNN(BaseMPDRNN):
    def __init__(self):
        super().__init__()

        # Create timestamp as the program begins to execute.
        timestamp = create_timestamp()
        colorama.init()

        penalty_term = self.cfg.get('penalty')
        rcond = self.cfg.get("rcond").get(self.cfg.get("method"))
        self.activation = self.cfg.get('activation')
        self.gen_ds_cfg = general_dataset_configs(self.cfg.get("dataset_name"))
        drnn_config = drnn_paths_config(self.dataset_name)

        self.filename = (
            os.path.join(
                drnn_config.get("mpdrnn").get("path_to_results"),
                f"{timestamp}_{self.dataset_name}_dataset_{self.method}_method_{penalty_term}"
                f"_penalty_{rcond:.4f}_rcond.xlsx"
            )
        )

        self.hyperparam_config = {
            "rcond": rcond,
            "penalty_term": penalty_term,
            "neurons": get_num_of_neurons(self.cfg, self.method)
        }

    def main(self) -> None:
        """
        Executes the main process of training and evaluating models for a specified number of tests.

        This method performs the following steps for each test iteration:
        1. Creates and trains an initial model.
        2. Creates and trains a subsequent model.
        3. Creates and trains a final model.
        4. Collects and logs metrics from all models.
        5. Saves the metrics to an Excel file.

        Returns:
            None
        """

        training_time = []

        for i in tqdm(range(self.cfg.get('number_of_tests')), desc=colorama.Fore.CYAN + "Process"):
            # Initial Model
            net_cfg = (
                self.get_network_config(
                    network_type="MultiPhaseDeepRandomizedNeuralNetworkBase",
                    config=self.hyperparam_config
                )
            )

            self.initial_model = (
                ModelFactory.create(
                    network_type="MultiPhaseDeepRandomizedNeuralNetworkBase",
                    network_cfg=net_cfg
                )
            )

            self.initial_model, initial_model_training_metrics, initial_model_testing_metrics = (
                self.model_training_and_evaluation(
                    model=self.initial_model,
                    eval_set=self.test_loader,
                    weights=self.initial_model.beta_weights,
                    num_hidden_layers=1,
                    verbose=True
                )
            )

            training_time.append(self.initial_model.train_ith_layer.execution_time)

            # Subsequent Model
            net_cfg = (
                self.get_network_config(
                    network_type="MultiPhaseDeepRandomizedNeuralNetworkSubsequent",
                    config=self.hyperparam_config
                )
            )

            self.subsequent_model = (
                ModelFactory.create(
                    network_type="MultiPhaseDeepRandomizedNeuralNetworkSubsequent",
                    network_cfg=net_cfg
                )
            )

            self.subsequent_model, subsequent_model_training_metrics, subsequent_model_testing_metrics = (
                self.model_training_and_evaluation(
                    model=self.subsequent_model,
                    eval_set=self.test_loader,
                    weights=[self.subsequent_model.extended_beta_weights,
                             self.subsequent_model.gamma_weights],
                    num_hidden_layers=2,
                    verbose=True
                )
            )

            training_time.append(self.subsequent_model.train_ith_layer.execution_time)

            # Final Model
            net_cfg = (
                self.get_network_config(
                    network_type="MultiPhaseDeepRandomizedNeuralNetworkFinal",
                    config=self.hyperparam_config
                )
            )

            final_model = (
                ModelFactory.create(
                    network_type="MultiPhaseDeepRandomizedNeuralNetworkFinal",
                    network_cfg=net_cfg
                )
            )

            final_model, final_model_training_metrics, final_model_testing_metrics = (
                self.model_training_and_evaluation(
                    model=final_model,
                    eval_set=self.test_loader,
                    weights=[final_model.extended_beta_weights,
                             final_model.extended_gamma_weights,
                             final_model.delta_weights],
                    num_hidden_layers=3,
                    verbose=True
                )
            )

            training_time.append(final_model.train_ith_layer.execution_time)

            metrics = (
                reorder_metrics_lists(
                    train_metrics=final_model_training_metrics,
                    test_metrics=final_model_testing_metrics,
                    training_time_list=training_time
                )
            )
            insert_data_to_excel(self.filename, self.cfg.get("dataset_name"), i + 2, metrics)
            training_time.clear()

        average_columns_in_excel(self.filename)


if __name__ == "__main__":
    try:
        mpdrnn = MPDRNN()
        mpdrnn.main()
    except KeyboardInterrupt as kie:
        logging.error(f"{kie}")