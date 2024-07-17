import colorama
import logging
import torch

from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score
from torchinfo import summary
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import general_dataset_configs, fcnn_paths_configs
from nn.models.fcnn_model import FCNN
from nn.dataloaders.npz_dataloader import NpzDataset
from utils.utils import (setup_logger, device_selector, load_config_json,
                         find_latest_file_in_latest_directory, plot_confusion_matrix_fcnn)


class EvalFCNN:
    def __init__(self):
        # Basic setup
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
        self.train_loader, self.test_loader = (
            self.create_train_test_datasets(file_path)
        )

        # Setup device
        self.device = (
            device_selector(preferred_device="cpu")
        )

        # Load the model
        self.model = (
            FCNN(input_size=gen_ds_cfg.get("num_features"),
                 hidden_size=self.cfg.get("hidden_neurons").get(self.cfg.get("dataset_name")),
                 output_size=gen_ds_cfg.get("num_classes")).to(self.device)
        )
        summary(self.model, input_size=(gen_ds_cfg.get("num_features"),), device=self.device)

        checkpoint = find_latest_file_in_latest_directory(fcnn_ds_cfg.get("fcnn_saved_weights"))
        self.model.load_state_dict(torch.load(checkpoint))
        self.model = self.model.to(self.device)

        self.train_accuracy = None
        self.test_accuracy = None
        self.train_precision = None
        self.test_precision = None
        self.train_recall = None
        self.test_recall = None
        self.train_f1sore = None
        self.test_f1sore = None

    def create_train_test_datasets(self, file_path):
        train_dataset = NpzDataset(file_path, operation="train")
        test_dataset = NpzDataset(file_path, operation="test")

        train_loader = (
            DataLoader(dataset=train_dataset,
                       batch_size=self.cfg.get("batch_size").get(self.cfg.get("dataset_name")),
                       shuffle=False)
        )

        test_loader = (
            DataLoader(dataset=test_dataset,
                       batch_size=self.cfg.get("batch_size").get(self.cfg.get("dataset_name")),
                       shuffle=False)
        )

        return train_loader, test_loader

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

        # Using getattr and setattr to dynamically set attributes
        setattr(self, f"{operation}_accuracy", accuracy)
        setattr(self, f"{operation}_precision", precision)
        setattr(self, f"{operation}_recall", recall)
        setattr(self, f"{operation}_f1sore", f1sore)
        setattr(self, f"{operation}_cm", cm)

        plot_confusion_matrix_fcnn(cm, operation, self.class_labels, self.cfg.get("dataset_name"))
        logging.info(f"{operation} accuracy: {accuracy:.4f}")
        logging.info(f"{operation} precision: {precision:.4f}")
        logging.info(f"{operation} recall: {recall:.4f}")
        logging.info(f"{operation} F1-score: {f1sore:.4f}")

    def main(self):
        self.evaluate_model(self.train_loader, "train")
        self.evaluate_model(self.test_loader, "test")


if __name__ == '__main__':
    eval_fcnn = EvalFCNN()
    eval_fcnn.main()
