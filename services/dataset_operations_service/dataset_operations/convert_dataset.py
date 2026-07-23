import logging
import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

from config.dataset_config import general_dataset_configs, VALID_DATASETS
from utils.common import setup_logger


def all_elements_numeric(nested_list):
    """Checks recursively whether all items in a nested list are numeric."""
    for item in nested_list:
        if isinstance(item, list):
            if not all_elements_numeric(item):
                return False
        else:
            if not str(item).isnumeric():
                try:
                    float(item)
                except ValueError:
                    return False
    return True


def split_dataset(dataset_name: str, split_ratio: list = None):
    """
    Splits and converts the dataset into Train, Validation, and Test sets,
    then saves it to a compressed .npz file.

    Args:
        dataset_name (str): Name of the dataset to process.
        split_ratio (list): Ratios for [train, validation, test]. Defaults to [0.7, 0.15, 0.15].
    """
    setup_logger()

    if split_ratio is None:
        split_ratio = [0.7, 0.15, 0.15]

    if len(split_ratio) != 3 or not np.isclose(sum(split_ratio), 1.0):
        raise ValueError("split_ratio must contain 3 elements that sum up to 1.0 (e.g. [0.7, 0.15, 0.15])")

    path_to_dataset = general_dataset_configs(dataset_name).get("dataset_file")
    size_of_dataset = general_dataset_configs(dataset_name).get("dataset_size")
    num_features = general_dataset_configs(dataset_name).get("num_features")

    try:
        with open(path_to_dataset, "r") as file:
            lines = file.readlines()

        labels = []
        features = []
        for line in tqdm(lines, desc=f"Reading {dataset_name}"):
            split = line.strip().split(',')

            if dataset_name in ["connect4", "isolete", "musk2", "optdigits", "page_blocks", "satimages",
                                "shuttle", "spambase", "usps", "wall", "waveform"]:
                labels.append(split[-1])
                features.append(split[:-1])
            elif dataset_name in ["letter", "mnist", "mnist_fashion", "segment"]:
                labels.append(split[0])
                features.append(split[1:])
            else:
                raise ValueError(f"Unsupported dataset name: {dataset_name}")

        label_encoder = OneHotEncoder(sparse_output=False)
        encoded_labels = label_encoder.fit_transform(np.array(labels).reshape(-1, 1))

        if all_elements_numeric(features):
            reshaped_features = np.array(features, dtype=np.float32)
            reshaped_features = reshaped_features.reshape((size_of_dataset, num_features))
        else:
            encoder = LabelEncoder()
            reshaped_features = []
            for feature in tqdm(features, desc="Encoding non-numeric features"):
                processed_feature = [0 if val == ' ?' else val for val in feature]
                reshaped_features.append(encoder.fit_transform(processed_feature))

        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(reshaped_features)

        size_of_test_subset = int(size_of_dataset * split_ratio[2])

        train_valid_x, test_x, train_valid_y, test_y = train_test_split(
            normalized_features,
            encoded_labels,
            test_size=size_of_test_subset,
            random_state=1234
        )


        validation_ratio = split_ratio[1] / (split_ratio[0] + split_ratio[1])

        train_x, valid_x, train_y, valid_y = train_test_split(
            train_valid_x,
            train_valid_y,
            test_size=validation_ratio,
            random_state=42
        )

        logging.info(
            f"[{dataset_name}] Train set size: {train_x.shape}, {train_x.shape[0] / size_of_dataset:.2%}"
        )
        logging.info(
            f"[{dataset_name}] Validation set size: {valid_x.shape}, {valid_x.shape[0] / size_of_dataset:.2%}"
        )
        logging.info(
            f"[{dataset_name}] Test set size: {test_x.shape}, {test_x.shape[0] / size_of_dataset:.2%}"
        )

        file_save_name = general_dataset_configs(dataset_name).get("cached_dataset_file")
        np.savez(
            file_save_name,
            train_x=train_x,
            valid_x=valid_x,
            test_x=test_x,
            train_y=train_y,
            valid_y=valid_y,
            test_y=test_y
        )

        logging.info(f"Successfully saved dataset to: {file_save_name}")

    except FileNotFoundError:
        logging.error(f"Dataset raw file not found at path: {path_to_dataset}")


def main():
    for dataset in VALID_DATASETS:
        split_dataset(dataset, split_ratio=[0.7, 0.15, 0.15])


if __name__ == "__main__":
    main()