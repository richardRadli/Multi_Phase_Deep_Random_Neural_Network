import logging
import torch

from torch.utils.data import DataLoader

from config.config import BWELMConfig
from config.dataset_config import bwelm_dataset_configs
from first_phase import FirstPhase
from nn.dataloader.npy_dataloader import NpyDataset
from utils.utils import display_dataset_info, setup_logger


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++ M U L T I   P H A S E   D E E P   R A N D O M I Z E   N E U R A L   N E T W O R K +++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class BackwardELM:
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- __I N I T__ ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        # Initialize paths and settings
        setup_logger()
        self.cfg = BWELMConfig().parse()
        self.gen_ds_cfg = bwelm_dataset_configs(self.cfg)
        display_dataset_info(self.gen_ds_cfg)

        if self.cfg.seed:
            torch.manual_seed(42)

        self.input_nodes = self.gen_ds_cfg.get("num_features")
        self.hidden_nodes = bwelm_dataset_configs(self.cfg).get("neurons")
        self.sigma = bwelm_dataset_configs(self.cfg).get("sigma")

        # Load data
        file_path = bwelm_dataset_configs(self.cfg).get("cached_dataset_file")
        train_dataset = NpyDataset(file_path, operation="train")
        test_dataset = NpyDataset(file_path, operation="test")
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- M A I N -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def main(self) -> None:
        """
        The main function that orchestrates the training and evaluation process of the ELM model.

        :return: None
        """
        init_layer = FirstPhase(n_input_nodes=self.input_nodes,
                                n_hidden_nodes=self.hidden_nodes,
                                sigma=self.sigma,
                                train_loader=self.train_loader,
                                test_loader=self.test_loader)
        init_layer.main()


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- __M A I N__ -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        bwelm = BackwardELM()
        bwelm.main()
    except KeyboardInterrupt as kie:
        logging.error(f'{kie}')
