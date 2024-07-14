import colorama
import logging
import os
import torch

from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score
from torchinfo import summary
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import general_dataset_configs, fcnn_paths_configs
from nn.models.fcnn_model import FCNN
from nn.dataloaders.npz_dataloader import NpzDataset
from utils.utils import (create_timestamp, setup_logger, use_gpu_if_available, load_config_json,
                         find_latest_file_in_latest_directory, plot_confusion_matrix_fcnn)


class EvalFCNN:
    def __init__(self):
        # Basic setup
        timestamp = create_timestamp()
        colorama.init()
        setup_logger()

        self.cfg = (
            load_config_json(json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_fcnn"),
                             json_filename=JSON_FILES_PATHS.get_data_path("config_fcnn"))
        )

        gen_ds_cfg = (
            general_dataset_configs(self.cfg.get("dataset_name"))
        )

        self.class_labels = gen_ds_cfg.get("class_labels")

        fcnn_ds_cfg = (
            fcnn_paths_configs(self.cfg.get("dataset_name"))
        )

        # Load the dataset
        file_path = (
            general_dataset_configs(self.cfg.get('dataset_name')).get("cached_dataset_file")
        )
        self.train_loader, _, self.test_loader = (
            self.create_train_test_datasets(file_path)
        )

        # Setup device
        self.device = (
            use_gpu_if_available()
        )

        # Load the model
        self.model = (
            FCNN(input_size=gen_ds_cfg.get("num_features"),
                 hidden_size=gen_ds_cfg.get("eq_neurons")[0],
                 output_size=gen_ds_cfg.get("num_classes")).to(self.device)
        )
        summary(self.model, input_size=(gen_ds_cfg.get("num_features"),))

        checkpoint = find_latest_file_in_latest_directory(fcnn_ds_cfg.get("fcnn_saved_weights"))
        self.model.load_state_dict(torch.load(checkpoint))
        self.model = self.model.to(self.device)

        self.save_path = os.path.join(fcnn_ds_cfg.get("saved_results"), timestamp)
        os.makedirs(self.save_path, exist_ok=True)

        self.training_acc = []
        self.testing_acc = []
        self.training_prec = []
        self.testing_prec = []
        self.training_recall = []
        self.testing_recall = []
        self.training_f1_score = []
        self.testing_f1_score = []

    def create_train_test_datasets(self, file_path):
        full_train_dataset = NpzDataset(file_path, operation="train")
        test_dataset = NpzDataset(file_path, operation="test")

        train_size = int(self.cfg.get("valid_size") * len(full_train_dataset))
        valid_size = len(full_train_dataset) - train_size

        train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, valid_size])

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.get("batch_size"), shuffle=False)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.cfg.get("batch_size"), shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.cfg.get("batch_size"), shuffle=False)

        return train_loader, valid_loader, test_loader

    def evaluate_model(self, dataloader, operation):
        self.model.eval()

        total_samples = 0
        correct_predictions = 0

        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch_data, batch_labels in tqdm(dataloader, total=len(dataloader), desc=f"Evaluating {operation} set"):
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                output = self.model(batch_data)

                predicted_labels = torch.argmax(output, 1)
                ground_truth_labels = torch.argmax(batch_labels, 1)

                correct_predictions += (predicted_labels == ground_truth_labels).sum().item()
                total_samples += batch_labels.size(0)

                all_preds.extend(predicted_labels.cpu().numpy())
                all_labels.extend(ground_truth_labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1sore = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        plot_confusion_matrix_fcnn(cm, operation, self.class_labels, self.cfg.get("dataset_name"))
        logging.info(f"{operation} accuracy: {accuracy:.4f}")
        logging.info(f"{operation} precision: {precision:.4f}")
        logging.info(f"{operation} recall: {recall:.4f}")
        logging.info(f"{operation} F1-score: {f1sore:.4f}")

        self.training_acc.append(accuracy) if operation == "train" else self.testing_acc.append(accuracy)
        self.training_prec.append(precision) if operation == "train" else self.testing_prec.append(precision)
        self.training_recall.append(recall) if operation == "train" else self.testing_recall.append(recall)

    def main(self):
        self.evaluate_model(self.train_loader, "train")
        self.evaluate_model(self.test_loader, "test")


if __name__ == '__main__':
    eval_fcnn = EvalFCNN()
    eval_fcnn.main()
