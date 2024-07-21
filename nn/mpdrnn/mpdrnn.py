import colorama
import os
import torch

from nn.dataloaders.npz_dataloader import NpzDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import general_dataset_configs, drnn_paths_config
from nn.models.mpdrnn_model import (MultiPhaseDeepRandomizedNeuralNetworkBase,
                                    MultiPhaseDeepRandomizedNeuralNetworkSubsequent,
                                    MultiPhaseDeepRandomizedNeuralNetworkFinal)
from utils.utils import (average_columns_in_excel, create_timestamp, insert_data_to_excel,
                         load_config_json, reorder_metrics_lists)


class MPDRNN:
    def __init__(self):
        # Create timestamp as the program begins to execute.
        timestamp = create_timestamp()

        # Initialize paths and settings
        self.cfg = (
            load_config_json(json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_mpdrnn"),
                             json_filename=JSON_FILES_PATHS.get_data_path("config_mpdrnn"))
        )
        dataset_name = self.cfg.get("dataset_name")
        gen_ds_cfg = general_dataset_configs(dataset_name)
        drnn_config = drnn_paths_config(dataset_name)

        self.filename = (
            os.path.join(
                drnn_config.get("mpdrnn").get("path_to_results"),
                f"{timestamp}_{dataset_name}_dataset_{self.cfg.get('method')}_method_"
                f"{self.cfg.get('penalty').get(self.cfg.get('dataset_name'))}_penalty.xlsx")
        )

        if self.cfg.get("method") not in ["BASE", "EXP_ORT", "EXP_ORT_C"]:
            raise ValueError(f"Wrong method was given: {self.cfg.get('method')}")

        if self.cfg.get("seed"):
            torch.manual_seed(1234)

        # Load neurons
        self.first_layer_num_data = gen_ds_cfg.get("num_train_data")
        self.first_layer_input_nodes = gen_ds_cfg.get("num_features")
        self.first_layer_hidden_nodes = self.get_num_of_neurons(self.cfg.get("method"), dataset_name)
        self.first_layer_output_nodes = gen_ds_cfg.get("num_classes")

        file_path = general_dataset_configs(dataset_name).get("cached_dataset_file")
        self.train_loader, _, self.test_loader = (
            self.create_train_valid_test_datasets(file_path)
        )

        colorama.init()

    def get_num_of_neurons(self, method, dataset_name):
        num_neurons = {
            "BASE": self.cfg.get("eq_neurons").get(dataset_name),
            "EXP_ORT": self.cfg.get("exp_neurons").get(dataset_name),
            "EXP_ORT_C": self.cfg.get("exp_neurons").get(dataset_name),
        }
        return num_neurons[method]

    @staticmethod
    def create_train_valid_test_datasets(file_path):
        train_dataset = NpzDataset(file_path, operation="train")
        valid_dataset = NpzDataset(file_path, operation="valid")
        test_dataset = NpzDataset(file_path, operation="test")

        train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset), shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

        return train_loader, valid_loader, test_loader

    def model_training_and_evaluation(self, model, weights, num_hidden_layers, verbose):
        model.train_layer(self.train_loader)

        training_metrics = model.predict_and_evaluate(
            dataloader=self.train_loader,
            operation="train",
            layer_weights=weights,
            num_hidden_layers=num_hidden_layers,
            verbose=verbose
        )

        testing_metrics = model.predict_and_evaluate(
            dataloader=self.test_loader,
            operation="test",
            layer_weights=weights,
            num_hidden_layers=num_hidden_layers,
            verbose=verbose
        )

        return model, training_metrics, testing_metrics

    def main(self):
        training_time = []

        for i in tqdm(range(self.cfg.get('number_of_tests')), desc=colorama.Fore.CYAN + "Process"):
            initial_model = (
                MultiPhaseDeepRandomizedNeuralNetworkBase(
                    num_data=self.first_layer_num_data,
                    num_features=self.first_layer_input_nodes,
                    hidden_nodes=self.first_layer_hidden_nodes,
                    output_nodes=self.first_layer_output_nodes,
                    activation_function=self.cfg.get('activation'),
                    method=self.cfg.get("method"),
                    penalty_term=self.cfg.get("penalty").get(self.cfg.get("dataset_name"))
                )
            )

            initial_model, initial_model_training_metrics, initial_model_testing_metrics = (
                self.model_training_and_evaluation(model=initial_model,
                                                   weights=initial_model.beta_weights,
                                                   num_hidden_layers=1,
                                                   verbose=True)
            )

            training_time.append(initial_model.train_ith_layer.execution_time)

            # Subsequent Model
            subsequent_model = (
                MultiPhaseDeepRandomizedNeuralNetworkSubsequent(base_instance=initial_model,
                                                                mu=self.cfg.get("mu"),
                                                                sigma=self.cfg.get("sigma"))
            )

            subsequent_model, subsequent_model_training_metrics, subsequent_model_testing_metrics = (
                self.model_training_and_evaluation(model=subsequent_model,
                                                   weights=[subsequent_model.extended_beta_weights,
                                                            subsequent_model.gamma_weights],
                                                   num_hidden_layers=2,
                                                   verbose=True)
            )

            training_time.append(subsequent_model.train_ith_layer.execution_time)

            final_model = (
                MultiPhaseDeepRandomizedNeuralNetworkFinal(subsequent_instance=subsequent_model,
                                                           mu=self.cfg.get("mu"),
                                                           sigma=self.cfg.get("sigma"))
            )

            final_model, final_model_training_metrics, final_model_testing_metrics = (
                self.model_training_and_evaluation(model=final_model,
                                                   weights=[final_model.extended_beta_weights,
                                                            final_model.extended_gamma_weights,
                                                            final_model.delta_weights],
                                                   num_hidden_layers=3,
                                                   verbose=True)
            )

            training_time.append(final_model.train_ith_layer.execution_time)

            metrics = reorder_metrics_lists(train_metrics=final_model_training_metrics,
                                            test_metrics=final_model_testing_metrics if final_model_testing_metrics[1] >
                                                                                        subsequent_model_testing_metrics[
                                                                                            1] else subsequent_model_testing_metrics,
                                            training_time_list=training_time)
            insert_data_to_excel(self.filename, self.cfg.get("dataset_name"), i + 2, metrics, "mpdrnn")
            training_time.clear()

        average_columns_in_excel(self.filename)


if __name__ == "__main__":
    mpdrnn = MPDRNN()
    mpdrnn.main()
