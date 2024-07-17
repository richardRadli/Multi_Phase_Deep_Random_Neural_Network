import colorama
import os
import logging
import torch

from nn.dataloaders.npz_dataloader import NpzDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import general_dataset_configs, drnn_paths_config
from nn.models.mpdrnn_model import (MultiPhaseDeepRandomizedNeuralNetworkBase,
                                    MultiPhaseDeepRandomizedNeuralNetworkSubsequent,
                                    MultiPhaseDeepRandomizedNeuralNetworkFinal)
from utils.utils import (average_columns_in_excel, create_timestamp, insert_data_to_excel, setup_logger,
                         load_config_json)


class MPDRNN:
    def __init__(self):
        # Create timestamp as the program begins to execute.
        timestamp = create_timestamp()

        # Initialize paths and settings
        setup_logger()
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
                f"{timestamp}_sp_{self.cfg.get('subset_percentage')}_pm_{self.cfg.get('pruning_method')}.xlsx")
        )

        if self.cfg.get("method") not in ["BASE", "EXP_ORT", "EXP_ORT_C"]:
            raise ValueError(f"Wrong method was given: {self.cfg.get('method')}")

        if self.cfg.get("seed"):
            torch.manual_seed(1234)

        # Load neurons
        self.first_layer_num_data = gen_ds_cfg.get("num_train_data")
        self.first_layer_input_nodes = gen_ds_cfg.get("num_features")
        self.first_layer_hidden_nodes = self.get_num_of_neurons(self.cfg.get("method"), dataset_name)[0]
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

    def initial_model_training_and_eval(self, initial_model):
        initial_model.train_first_layer(self.train_loader)
        tr1acc = initial_model.predict_and_evaluate(dataloader=self.train_loader,
                                                    operation="train",
                                                    layer_weights=initial_model.beta_weights,
                                                    num_hidden_layers=1,
                                                    verbose=True)
        te1acc = initial_model.predict_and_evaluate(dataloader=self.test_loader,
                                                    operation="test",
                                                    layer_weights=initial_model.beta_weights,
                                                    num_hidden_layers=1,
                                                    verbose=True)

        return initial_model, tr1acc, te1acc

    def subsequent_model_training_and_eval(self, subsequent_model):
        subsequent_model.train_second_layer(self.train_loader)
        tr2acc = subsequent_model.predict_and_evaluate(self.train_loader, "train")
        te2acc = subsequent_model.predict_and_evaluate(self.test_loader, "test")

        return subsequent_model, tr2acc, te2acc

    def final_model_training_and_eval(self, final_model):
        final_model.train_third_layer(self.train_loader)
        final_model.predict_and_evaluate(self.train_loader, "train")
        final_model.predict_and_evaluate(self.test_loader, "test")
        return final_model

    def main(self):
        # Load data
        accuracies = []

        for i in tqdm(range(self.cfg.get('number_of_tests')), desc=colorama.Fore.CYAN + "Process"):
            initial_model = MultiPhaseDeepRandomizedNeuralNetworkBase(num_data=self.first_layer_num_data,
                                                                      num_features=self.first_layer_input_nodes,
                                                                      hidden_nodes=self.first_layer_hidden_nodes,
                                                                      output_nodes=self.first_layer_output_nodes,
                                                                      activation_function=self.cfg.get('activation'))

            # Train and evaluate the model
            initial_model, initial_model_training_acc, initial_model_testing_acc = (
                self.initial_model_training_and_eval(initial_model)
            )

            # Subsequent Model
            subsequent_model = (
                MultiPhaseDeepRandomizedNeuralNetworkSubsequent(base_instance=initial_model,
                                                                mu=self.cfg.get("mu"),
                                                                sigma=self.cfg.get("sigma"))
            )
            subsequent_model, subsequent_model_training_acc, subsequent_model_testing_acc = (
                self.subsequent_model_training_and_eval(subsequent_model)
            )

            final_model = MultiPhaseDeepRandomizedNeuralNetworkFinal(base_instance=subsequent_model,
                                                                     mu=self.cfg.get("mu"),
                                                                     sigma=self.cfg.get("sigma"))
            self.final_model_training_and_eval(final_model)


if __name__ == "__main__":
    mpdrnn = MPDRNN()
    mpdrnn.main()
