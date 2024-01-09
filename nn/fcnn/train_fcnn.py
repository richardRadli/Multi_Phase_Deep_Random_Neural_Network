import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from config.config import FCNNConfig
from config.dataset_config import general_dataset_configs, fcnn_dataset_configs
from dataset_operations.load_dataset import load_data_fcnn
from utils.utils import create_timestamp, setup_logger, measure_execution_time_fcnn
from model import CustomELMModel


class FCNN:
    def __init__(self):
        # Create time stamp
        self.timestamp = create_timestamp()

        # Set up logger
        setup_logger()

        # Set up config
        self.fcnn_cfg = FCNNConfig().parse()

        # Set up paths
        gen_ds_cfg = general_dataset_configs(self.fcnn_cfg)
        fcnn_ds_cfg = fcnn_dataset_configs(self.fcnn_cfg)

        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load dataset
        self.train_loader, self.valid_loader, self.test_loader = load_data_fcnn(gen_ds_cfg, self.fcnn_cfg)

        # Set up model
        self.model = CustomELMModel()
        summary(self.model, input_size=(gen_ds_cfg.get("num_features"),))

        # Define your loss function
        self.loss_fn = nn.MSELoss()

        # Define your optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.fcnn_cfg.learning_rate)

        # Tensorboard
        tensorboard_log_dir = os.path.join(fcnn_ds_cfg.get("logs"), self.timestamp)
        if not os.path.exists(tensorboard_log_dir):
            os.makedirs(tensorboard_log_dir)
        self.writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

        # Create save directory
        self.save_path = os.path.join(fcnn_ds_cfg.get("fcnn_saved_weights"), self.timestamp)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Initialize aux variables
        self.training_acc = []
        self.testing_acc = []
        self.training_time = []

    def train_loop(self, batch_data, batch_labels, train_losses):
        self.optimizer.zero_grad()
        batch_data = batch_data.to(self.device)
        batch_labels = batch_labels.to(self.device)

        outputs = self.model(batch_data)
        t_loss = self.loss_fn(outputs, batch_labels)
        t_loss.backward()
        self.optimizer.step()

        train_losses.append(t_loss.item())

    def valid_loop(self, batch_data, batch_labels, valid_losses):
        batch_data = batch_data.to(self.device)
        batch_labels = batch_labels.to(self.device)

        outputs = self.model(batch_data)
        v_loss = self.loss_fn(outputs, batch_labels)

        valid_losses.append(v_loss.item())

    @measure_execution_time_fcnn
    def train(self):
        best_valid_loss = float('inf')
        best_model_path = None
        epochs_without_improvement = 0

        # To track the training loss as the model trains
        train_losses = []

        # To track the validation loss as the model trains
        valid_losses = []

        for epoch in tqdm(range(self.fcnn_cfg.epochs), desc="Training", total=self.fcnn_cfg.epochs):
            self.model.train()
            for batch_data, batch_labels in tqdm(self.train_loader, total=len(self.train_loader)):
                self.train_loop(batch_data, batch_labels, train_losses)

            for batch_data, batch_labels in tqdm(self.valid_loader, total=len(self.valid_loader)):
                self.valid_loop(batch_data, batch_labels, valid_losses)

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)

            logging.info(f'\ntrain_loss: {train_loss:.5f} ' + f'valid_loss: {valid_loss:.5f}')

            # Record to tensorboard
            self.writer.add_scalars("Loss", {"train": train_loss, "validation": valid_loss}, epoch)

            # Clear lists to track next epoch
            train_losses.clear()
            valid_losses.clear()

            # Early stopping check
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_without_improvement = 0
                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = os.path.join(self.save_path, "epoch_" + str(epoch) + ".pt")
                torch.save(self.model.state_dict(), best_model_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.fcnn_cfg.patience:
                    logging.info("Early stopping: No improvement for {} epochs".format(self.fcnn_cfg.patience))
                    break

        # Close and flush SummaryWriter
        self.writer.close()
        self.writer.flush()

    def evaluate(self, data_loader, operation: str):
        self.model.eval()
        total_samples = 0
        correct_predictions = 0

        with torch.no_grad():
            for batch_data, batch_labels in data_loader:
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

    def main(self):
        self.train()
        self.evaluate(self.train_loader, operation="train")
        self.evaluate(self.test_loader, operation="test")


if __name__ == "__main__":
    try:
        fcnn = FCNN()
        fcnn.main()
    except KeyboardInterrupt as kie:
        logging.error(f'{kie}')
