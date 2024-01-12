import colorama
import logging
import os
import pandas as pd
import torch

from torchsummary import summary
from tqdm import tqdm

from config.config import DatasetConfig, FCNNConfig
from config.dataset_config import general_dataset_configs, fcnn_dataset_configs
from dataset_operations.load_dataset import load_data_fcnn
from model import CustomELMModel
from utils.utils import (create_timestamp, find_latest_file_in_latest_directory, measure_execution_time_fcnn,
                         setup_logger, use_gpu_if_available)


class EvalFCNN:
    def __init__(self):
        # Create time stamp
        self.timestamp = create_timestamp()

        # Set up colorama
        colorama.init()

        # Set up logger
        setup_logger()

        # Set up device
        self.device = use_gpu_if_available()

        # Set up config
        dataset_cfg = DatasetConfig().parse()
        self.fcnn_cfg = FCNNConfig().parse()

        # Set up paths
        gen_ds_cfg = general_dataset_configs(dataset_cfg)
        fcnn_ds_cfg = fcnn_dataset_configs(dataset_cfg)

        # Load dataset
        self.train_loader, _, self.test_loader = load_data_fcnn(gen_ds_cfg, self.fcnn_cfg)

        # Set up model
        self.model = CustomELMModel()
        checkpoint_path = find_latest_file_in_latest_directory(fcnn_ds_cfg.get("fcnn_saved_weights"))
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model = self.model.to(self.device)
        summary(self.model, input_size=(gen_ds_cfg.get("num_features"),))

        # Create save directory
        self.save_path = os.path.join(fcnn_ds_cfg.get("saved_results"), self.timestamp)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Aux variables
        self.training_acc = []
        self.testing_acc = []

    @measure_execution_time_fcnn
    def evaluate(self, data_loader, operation: str) -> None:
        """
        Evaluate the model on the given dataset.

        Args:
            data_loader (DataLoader): DataLoader containing the evaluation dataset.
            operation (str): String indicating the type of evaluation ("train" or "test").

        Returns:
            None
        """

        self.model.eval()
        total_samples = 0
        correct_predictions = 0

        with torch.no_grad():
            for batch_data, batch_labels in tqdm(data_loader,
                                                 total=len(data_loader),
                                                 desc=colorama.Fore.LIGHTCYAN_EX + operation + " accuracy calculating"):
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_data)
                predicted_labels = torch.argmax(outputs, dim=1)
                correct_predictions += (predicted_labels == torch.argmax(batch_labels, dim=1)).sum().item()
                total_samples += batch_labels.size(0)

        accuracy = correct_predictions / total_samples

        logging.info(f'The {operation} accuracy is {accuracy:.4f}')

        if operation == "train":
            self.training_acc.append(accuracy)
        else:
            self.testing_acc.append(accuracy)

    def save_to_file(self) -> None:
        """
        Saves the results to a txt file.
        Returns:
            None
        """

        df = pd.DataFrame({"Training_accuracy": self.training_acc,
                           "Testing_accuracy": self.testing_acc})
        df.to_csv(os.path.join(self.save_path, "results.txt"), index=False)

    def main(self) -> None:
        """
        Executes the evaluation on the training and testing data.
        Returns:
            None
        """

        fcnn.evaluate(self.train_loader, operation="train")
        fcnn.evaluate(self.test_loader, operation="test")
        fcnn.save_to_file()


if __name__ == '__main__':
    fcnn = EvalFCNN()
    fcnn.main()
