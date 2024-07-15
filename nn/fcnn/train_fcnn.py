import colorama
import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import general_dataset_configs, fcnn_paths_configs
from nn.models.fcnn_model import FCNN
from nn.dataloaders.npz_dataloader import NpzDataset
from utils.utils import create_timestamp, setup_logger, use_gpu_if_available, load_config_json, measure_execution_time


class TrainFCNN:
    def __init__(self):
        # Basic setup
        timestamp = create_timestamp()
        colorama.init()
        setup_logger()

        self.cfg = (
            load_config_json(json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_fcnn"),
                             json_filename=JSON_FILES_PATHS.get_data_path("config_fcnn"))
        )

        if self.cfg.get("seed"):
            torch.manual_seed(1234)

        gen_ds_cfg = (
            general_dataset_configs(self.cfg.get("dataset_name"))
        )

        fcnn_ds_cfg = (
            fcnn_paths_configs(self.cfg.get("dataset_name"))
        )

        # Setup device
        self.device = (
            use_gpu_if_available()
        )

        # Load the model
        self.model = (
            FCNN(input_size=gen_ds_cfg.get("num_features"),
                 hidden_size=self.cfg.get("hidden_neurons"),
                 output_size=gen_ds_cfg.get("num_classes")).to(self.device)
        )

        summary(self.model, input_size=(gen_ds_cfg.get("num_features"),))

        # Load the dataset
        file_path = (
            general_dataset_configs(self.cfg.get('dataset_name')).get("cached_dataset_file")
        )
        self.train_loader, self.valid_loader, self.test_loader = (
            self.create_train_test_datasets(file_path)
        )

        # Define loss function
        self.criterion = (
            nn.CrossEntropyLoss()
        )

        # Define optimizer
        self.optimizer = (
            optim.Adam(self.model.parameters(),
                       lr=self.cfg.get("learning_rate"))
        )

        # Tensorboard
        tensorboard_log_dir = os.path.join(fcnn_ds_cfg.get("logs"), timestamp)
        if not os.path.exists(tensorboard_log_dir):
            os.makedirs(tensorboard_log_dir)
        self.writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

        # Create save directory
        self.save_path = os.path.join(fcnn_ds_cfg.get("fcnn_saved_weights"), timestamp)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def create_train_test_datasets(self, file_path):
        full_train_dataset = NpzDataset(file_path, operation="train")
        test_dataset = NpzDataset(file_path, operation="test")

        train_size = int(self.cfg.get("valid_size") * len(full_train_dataset))
        valid_size = len(full_train_dataset) - train_size

        train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, valid_size])

        train_loader = (
            DataLoader(dataset=train_dataset,
                       batch_size=self.cfg.get("batch_size"),
                       shuffle=False)
        )
        valid_loader = (
            DataLoader(dataset=valid_dataset,
                       batch_size=self.cfg.get("batch_size"),
                       shuffle=False)
        )
        test_loader = (
            DataLoader(dataset=test_dataset,
                       batch_size=self.cfg.get("batch_size"),
                       shuffle=False)
        )

        return train_loader, valid_loader, test_loader

    def train_loop(self, batch_data, batch_labels, train_losses):
        batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

        self.optimizer.zero_grad()

        output = self.model(batch_data)

        loss = self.criterion(output, batch_labels)
        loss.backward()
        self.optimizer.step()

        train_losses.append(loss.item())

    def valid_loop(self, batch_data, batch_labels, valid_losses):
        batch_data = batch_data.to(self.device)
        batch_labels = batch_labels.to(self.device)

        output = self.model(batch_data)

        t_loss = self.criterion(output, batch_labels)
        valid_losses.append(t_loss.item())

    @measure_execution_time
    def fit(self):
        best_valid_loss = float("inf")
        best_model_path = None
        epoch_without_improvement = 0

        train_losses = []
        valid_losses = []

        for epoch in tqdm(range(self.cfg.get("epochs"))):

            self.model.train()
            for batch_data, batch_labels in self.train_loader:
                self.train_loop(batch_data, batch_labels, train_losses)

            self.model.eval()
            for batch_data, batch_labels in self.valid_loader:
                self.valid_loop(batch_data, batch_labels, valid_losses)

            train_loss = np.mean(train_losses)
            valid_loss = np.mean(valid_losses)

            logging.info(f'\ntrain_loss: {train_loss:.4f} ' + f'valid_loss: {valid_loss:.4f}')

            self.writer.add_scalars("Loss", {"train": train_loss, "validation": valid_loss}, epoch)

            train_losses.clear()
            valid_losses.clear()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epoch_without_improvement = 0
                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = os.path.join(self.save_path, f"best_model_epoch_{epoch}.pt")
                torch.save(self.model.state_dict(), best_model_path)
                logging.info(f'New weights have been saved at epoch {epoch} with value of {best_valid_loss:.4f}')
            else:
                logging.warning(f"No new weights have been saved. Best valid loss was {best_valid_loss:.5f},\n "
                                f"current valid loss is {valid_loss:.5f}")
                epoch_without_improvement += 1
                if epoch_without_improvement >= self.cfg.get("patience"):
                    logging.info(f"Early stopping at epoch {epoch}")
                    break

        self.writer.close()
        self.writer.flush()


if __name__ == '__main__':
    try:
        train_fcnn = TrainFCNN()
        train_fcnn.fit()
    except KeyboardInterrupt as kie:
        logging.error(f'{kie}')
