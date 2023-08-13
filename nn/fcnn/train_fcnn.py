import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from elm.src.config.config import FCNNConfig
from elm.src.config.dataset_config import general_dataset_configs, fcnn_dataset_configs
from elm.src.utils.utils import create_timestamp, setup_logger
from model import CustomELMModel


class FCNN:
    def __init__(self):
        # Create time stamp
        self.timestamp = create_timestamp()

        # Set up logger
        setup_logger()

        # Set up config
        self.cfg = FCNNConfig().parse()

        #
        gen_ds_cfg = general_dataset_configs(self.cfg)
        fcnn_ds_cfg = fcnn_dataset_configs(self.cfg)

        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.training_acc = []
        self.testing_acc = []
        self.training_time = []

        # Load the fcnn_data directly
        data_file = gen_ds_cfg.get("cached_dataset_file")
        data = np.load(data_file, allow_pickle=True)
        train_data = torch.tensor(data[0], dtype=torch.float32)
        train_labels = torch.tensor(data[2], dtype=torch.float32)

        # Split fcnn_data into training and validation sets
        num_samples = len(train_data)
        num_train = int(self.cfg.train_size * num_samples)  # 80% for training
        self.train_data, self.valid_data = train_data[:num_train], train_data[num_train:]
        self.train_labels, self.valid_labels = train_labels[:num_train], train_labels[num_train:]

        self.test_data = torch.tensor(data[1], dtype=torch.float32)
        self.test_labels = torch.tensor(data[3], dtype=torch.float32)

        self.model = CustomELMModel()
        summary(self.model, input_size=(gen_ds_cfg.get("num_features"),))

        # Define your loss function
        self.loss_fn = nn.MSELoss()

        # Define your optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)

        # Tensorboard
        tensorboard_log_dir = os.path.join(fcnn_ds_cfg.get("logs"), self.timestamp)
        if not os.path.exists(tensorboard_log_dir):
            os.makedirs(tensorboard_log_dir)
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

        # Create save directory
        self.save_path = os.path.join(fcnn_ds_cfg.get("fcnn_saved_weights"), self.timestamp)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def train(self):
        best_valid_loss = float('inf')
        best_model_path = None
        epochs_without_improvement = 0

        # To track the training loss as the model trains
        train_losses = []

        # To track the validation loss as the model trains
        valid_losses = []

        for epoch in tqdm(range(self.cfg.epochs), desc="Training", total=self.cfg.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.train_data)
            t_loss = self.loss_fn(outputs, self.train_labels)
            t_loss.backward()
            self.optimizer.step()
            train_losses.append(t_loss.item())

            # Validate and check for early stopping
            valid_outputs = self.model(self.valid_data)
            v_loss = self.loss_fn(valid_outputs, self.valid_labels)
            valid_losses.append(v_loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)

            logging.info(f'train_loss: {train_loss:.5f} ' + f'valid_loss: {valid_loss:.5f}')

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
                if epochs_without_improvement >= self.cfg.patience:
                    logging.info("Early stopping: No improvement for {} epochs".format(self.cfg.patience))
                    break

        # Close and flush SummaryWriter
        self.writer.close()
        self.writer.flush()

    def evaluate(self, data, labels, operation: str):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            predicted_labels = torch.argmax(outputs, dim=1)
            correct_predictions = (predicted_labels == torch.argmax(labels, dim=1)).sum().item()
            total_samples = labels.size(0)

        accuracy = correct_predictions / total_samples

        logging.info(f'The {operation} accuracy is {accuracy:.4f}')

        self.training_acc.append(accuracy) if operation == "train" else self.testing_acc.append(accuracy)

    def main(self):
        self.train()
        self.evaluate(self.train_data, self.train_labels, operation="train")
        self.evaluate(self.test_data, self.test_labels, operation="test")


if __name__ == "__main__":
    try:
        fcnn = FCNN()
        fcnn.main()
    except KeyboardInterrupt as kie:
        logging.error(f'{kie}')
