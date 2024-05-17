import logging
import torch

from torch.utils.data import DataLoader

from additional_layers import AdditionalLayer
from config.config import MPDRNNConfig
from config.dataset_config import elm_general_dataset_configs
from inital_layer import InitialLayer
from nn.dataloader.npz_dataloader import NpzDataset
from utils.utils import create_dir, display_dataset_info, setup_logger


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++ M U L T I   P H A S E   D E E P   R A N D O M I Z E   N E U R A L   N E T W O R K +++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class MultiPhaseDeepRandomizedNeuralNetwork:
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- __I N I T__ ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        # Initialize paths and settings
        setup_logger()
        self.cfg = MPDRNNConfig().parse()
        self.gen_ds_cfg = elm_general_dataset_configs(self.cfg)
        display_dataset_info(self.gen_ds_cfg)

        if self.cfg.method not in ["BASE", "EXP_ORT", "EXP_ORT_C"]:
            raise ValueError(f"Wrong method was given: {self.cfg.method}")

        if self.cfg.seed:
            torch.manual_seed(42)

        self.method = self.cfg.method
        self.activation = self.cfg.activation
        self.first_layer_input_nodes = self.gen_ds_cfg.get("num_features")
        self.first_layer_hidden_nodes = self.get_num_of_neurons(self.method)[0]
        self.second_layer_hidden_nodes = self.get_num_of_neurons(self.method)[1]
        self.third_layer_hidden_nodes = self.get_num_of_neurons(self.method)[2]

        # Load data
        file_path = elm_general_dataset_configs(self.cfg).get("cached_dataset_file")
        train_dataset = NpzDataset(file_path, operation="train")
        test_dataset = NpzDataset(file_path, operation="test")
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

        self.directories = self.generate_directories()

    def generate_directories(self):
        directories_path = {}

        cm_root = self.gen_ds_cfg.get("path_to_cm")
        cm_dir = create_dir(
            root_dir=cm_root,
            method=self.cfg.method
        )

        metrics_root = self.gen_ds_cfg.get("path_to_metrics")
        metrics_dir = create_dir(
            root_dir=metrics_root,
            method=self.cfg.method
        )

        results_root = self.gen_ds_cfg.get("path_to_results")
        results_dir = create_dir(
            root_dir=results_root,
            method=self.cfg.method
        )

        directories_path["cm"] = cm_dir
        directories_path["metrics"] = metrics_dir
        directories_path["results"] = results_dir

        return directories_path

    def get_num_of_neurons(self, method):
        """

        Args:
            method:

        Returns:

        """

        num_neurons = {
            "BASE": self.gen_ds_cfg.get("eq_neurons"),
            "EXP_ORT": self.gen_ds_cfg.get("exp_neurons"),
            "EXP_ORT_C": self.gen_ds_cfg.get("exp_neurons"),
        }
        return num_neurons[method]

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- M A I N -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def main(self) -> None:
        """
        The main function that orchestrates the training and evaluation process of the ELM model.

        :return: None
        """

        init_layer = InitialLayer(n_input_nodes=self.first_layer_input_nodes,
                                  n_hidden_nodes=self.first_layer_hidden_nodes,
                                  train_loader=self.train_loader,
                                  test_loader=self.test_loader,
                                  activation=self.cfg.activation,
                                  method=self.method,
                                  phase_name="Phase 1",
                                  directory_path=self.directories)
        init_layer.main()
        second_layer = AdditionalLayer(previous_layer=init_layer,
                                       train_loader=self.train_loader,
                                       test_loader=self.test_loader,
                                       n_hidden_nodes=self.second_layer_hidden_nodes,
                                       mu=self.cfg.mu,
                                       sigma=self.cfg.sigma_layer_2,
                                       activation=self.cfg.activation,
                                       method=self.method,
                                       phase_name="Phase 2",
                                       directory_path=self.directories)
        second_layer.main()
        third_layer = AdditionalLayer(previous_layer=second_layer,
                                      train_loader=self.train_loader,
                                      test_loader=self.test_loader,
                                      n_hidden_nodes=self.third_layer_hidden_nodes,
                                      mu=self.cfg.mu,
                                      sigma=self.cfg.sigma_layer_3,
                                      activation=self.cfg.activation,
                                      method=self.method,
                                      phase_name="Phase 3",
                                      general_settings=self.gen_ds_cfg,
                                      config=self.cfg,
                                      directory_path=self.directories)
        third_layer.main()


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- __M A I N__ -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        mp_d_rnn = MultiPhaseDeepRandomizedNeuralNetwork()
        mp_d_rnn.main()
    except KeyboardInterrupt as kie:
        logging.error(f'{kie}')
