import colorama
import os
import logging

from tqdm import tqdm
from config.dataset_config import general_dataset_configs, drnn_paths_config
from nn.dev_drnn.base_class_dev_drnn import BaseDevDRNN
from nn.models.layer_selector import LayerFactory
from utils.utils import (average_columns_in_excel, create_timestamp, 
                         insert_data_to_excel, reorder_metrics_lists)


class DevDRNN(BaseDevDRNN):
    def __init__(self):
        super().__init__()

        timestamp = create_timestamp()
        colorama.init()

        penalty_term = self.cfg.get('penalty')
        rcond = self.cfg.get("rcond")
        self.activation = self.cfg.get('activation')
        self.gen_ds_cfg = general_dataset_configs(self.cfg.get("dataset_name"))
        drnn_config = drnn_paths_config(self.dataset_name)

        self.filename = (
            os.path.join(
                drnn_config.get("dev_drnn").get("path_to_results"),
                f"{timestamp}_{self.dataset_name}_dataset_{penalty_term}"
                f"_penalty_{rcond:.4f}_rcond.xlsx"
            )
        )

        self.hyperparam_config = {
            "rcond": rcond,
            "penalty_term": penalty_term,
            "neurons": self.cfg.get("exp_neurons")
        }

    def main(self) -> None:
        training_time = []
        
        for i in tqdm(range(self.cfg.get('number_of_tests')), desc=colorama.Fore.CYAN + "Process"):
            # First layer
            first_layer_cfg = (
            self.get_network_config(
                network_type="DevDeepRandomizedNeuralNetworkFirstLayer",
                config=self.hyperparam_config
            )
        )

            self.first_layer = (
                LayerFactory.create(
                    network_type="DevDeepRandomizedNeuralNetworkFirstLayer",
                    network_cfg=first_layer_cfg,
                    train_loader=self.train_loader
                )
            )

            self.first_layer, _, _ = (
                self.model_training_and_evaluation(
                    model=self.first_layer,
                    weights=self.first_layer.beta_weights,
                    num_hidden_layers=1,
                    verbose=True
                )
            )

            training_time.append(self.first_layer.train_ith_layer.execution_time)

            # Second layer
            second_layer_cfg = (
                self.get_network_config(
                    network_type="DevDeepRandomizedNeuralNetworkSecondLayer",
                    config=self.hyperparam_config
                )
            )
            self.second_layer = (
                LayerFactory.create(
                    network_type="DevDeepRandomizedNeuralNetworkSecondLayer",
                    network_cfg=second_layer_cfg
                )
            )

            self.second_layer, _, _ = (
                self.model_training_and_evaluation(
                    model=self.second_layer,
                    weights=[
                        self.second_layer.extended_beta_weights,
                        self.second_layer.gamma_weights
                    ],
                    num_hidden_layers=2,
                    verbose=True
                )
            )

            training_time.append(self.second_layer.train_ith_layer.execution_time)

            # Third layer
            third_layer_cfg = (
                self.get_network_config(
                    network_type="DevDeepRandomizedNeuralNetworkThirdLayer",
                    config=self.hyperparam_config
                )
            )

            third_layer = (
                LayerFactory.create(
                    network_type="DevDeepRandomizedNeuralNetworkThirdLayer",
                    network_cfg=third_layer_cfg
                )
            )

            third_layer, training_metrics, testing_metrics = (
                self.model_training_and_evaluation(
                    model=third_layer,
                    weights=[
                        third_layer.extended_beta_weights,
                        third_layer.extended_gamma_weights,
                        third_layer.delta_weights
                    ],
                    num_hidden_layers=3,
                    verbose=True
                )
            )

            training_time.append(third_layer.train_ith_layer.execution_time)

            metrics = (
                reorder_metrics_lists(
                    train_metrics=training_metrics,
                    test_metrics=testing_metrics,
                    training_time_list=training_time
                )
            )
            insert_data_to_excel(self.filename, self.cfg.get("dataset_name"), i + 2, metrics)
            training_time.clear()

        average_columns_in_excel(self.filename)


if __name__ == "__main__":
    try:
        dev_drnn = DevDRNN()
        dev_drnn.main()
    except KeyboardInterrupt as kie:
        logging.error(f"{kie}")
