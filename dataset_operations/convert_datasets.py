import logging
import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

from config.data_paths import ConfigFilePaths
from config.dataset_config import general_dataset_configs
from utils.utils import setup_logger, load_config_json


def all_elements_numeric(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            # Recursively check elements in nested lists
            if not all_elements_numeric(item):
                return False
        else:
            # Check if the individual item is numeric
            if not str(item).isnumeric():
                try:
                    float(item)
                except ValueError:
                    return False
    return True


def main(split_ratio):
    """

    Args:
        split_ratio: list, where elements denote train validation and test split ratio subsequently.

    Returns:
        Notes
    """

    setup_logger()

    cfg = (
        load_config_json(
            json_schema_filename=ConfigFilePaths().get_data_path("config_schema_ipmpdrnn"),
            json_filename=ConfigFilePaths().get_data_path("config_ipmpdrnn")
        )
    )

    dataset_name = cfg.get("dataset_name")

    path_to_dataset = general_dataset_configs(dataset_name).get("dataset_file")
    size_of_dataset = (general_dataset_configs(dataset_name).get("dataset_size"))
    num_features = general_dataset_configs(dataset_name).get("num_features")

    try:
        with open(path_to_dataset, "r") as file:
            lines = file.readlines()

        # Split the data into labels and features
        labels = []
        features = []
        for line in tqdm(lines):
            split = line.strip().split(',')
            # label at the end
            if dataset_name in ["connect4", "isolete", "iris", "musk2", "optdigits", "page_blocks", "satimages",
                                "shuttle", "spambase", "forest", "usps", "wall", "waveform"]:
                labels.append(split[-1])
                features.append(split[:-1])
            # label at the front
            elif dataset_name in ["letter", "mnist", "mnist_fashion", "segment"]:
                labels.append(split[0])
                features.append(split[1:])
            else:
                raise ValueError(f"Wrong dataset name {dataset_name}")

        label_encoder = OneHotEncoder(sparse_output=False)
        encoded_labels = label_encoder.fit_transform(np.array(labels).reshape(-1, 1))

        if all_elements_numeric(features):
            reshaped_features = np.array(features, dtype=np.float32)
            reshaped_features = reshaped_features.reshape((size_of_dataset, num_features))
        else:
            encoder = LabelEncoder()
            reshaped_features = []
            for feature in tqdm(features):
                reshaped_features.append(encoder.fit_transform(feature))

        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(reshaped_features)

        size_of_test_subset = int(size_of_dataset * split_ratio[2])

        train_valid_x, test_x, train_valid_y, test_y = (
            train_test_split(
                normalized_features,
                encoded_labels,
                test_size=size_of_test_subset,
                random_state=1234
            )
        )

        validation_ratio = split_ratio[1] / (split_ratio[0] + split_ratio[2])

        train_x, valid_x, train_y, valid_y = (
            train_test_split(
                train_valid_x,
                train_valid_y,
                test_size=validation_ratio,
                random_state=1234
            )
        )

        # Check the assertions
        logging.info(
            f"Train set size: {train_x.shape}, {train_x.shape[0] / size_of_dataset:.4f} percent of the dataset"
        )
        logging.info(
            f"Validation set size: {valid_x.shape}, {valid_x.shape[0] / size_of_dataset:.4f} percent of the dataset"
        )
        logging.info(
            f"Test set size: {test_x.shape}, {test_x.shape[0] / size_of_dataset:.4f} percent of the dataset"
        )

        file_save_name = general_dataset_configs(dataset_name).get("cached_dataset_file")
        np.savez(file_save_name,
                 train_x=train_x,
                 valid_x=valid_x,
                 test_x=test_x,
                 train_y=train_y,
                 valid_y=valid_y,
                 test_y=test_y)

        logging.info(f"Saved dataset to {file_save_name}")

    except FileNotFoundError:
        logging.error("File not found")


if __name__ == "__main__":
    main(split_ratio=[0.7, 0.15, 0.15])
