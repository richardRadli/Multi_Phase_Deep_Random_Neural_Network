import logging
import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

from config.data_paths import ConfigFilePaths
from config.dataset_config import general_dataset_configs
from utils.utils import setup_logger, load_config_json


def all_elements_numeric(nested_list):
    """

    Args:
        nested_list:

    Returns:

    """

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


def split_dataset(dataset_name):
    """
    Returns:
        Notes
    """

    setup_logger()

    cfg = (
        load_config_json(
            json_schema_filename=ConfigFilePaths().get_data_path("config_schema_mpdrnn"),
            json_filename=ConfigFilePaths().get_data_path("config_mpdrnn")
        )
    )

    path_to_dataset = general_dataset_configs(dataset_name).get("dataset_file")
    size_of_dataset = general_dataset_configs(dataset_name).get("dataset_size")
    num_features = general_dataset_configs(dataset_name).get("num_features")
    size_of_train_subset = general_dataset_configs(dataset_name).get("num_train_data")
    size_of_test_subset = general_dataset_configs(dataset_name).get("num_test_data")

    try:
        with open(path_to_dataset, "r") as file:
            lines = file.readlines()

        # Split the data into labels and features
        labels = []
        features = []
        for line in tqdm(lines):
            split = line.strip().split(',')
            # label at the end
            if dataset_name in ["connect4", "isolete", "musk2", "optdigits", "page_blocks", "shuttle", "usps"]:
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
                processed_feature = [0 if val == ' ?' else val for val in feature]
                reshaped_features.append(encoder.fit_transform(processed_feature))

        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(reshaped_features)

        indices = np.arange(len(normalized_features))

        if cfg.get("seed"):
            np.random.seed(42)
        np.random.shuffle(indices)

        shuffled_features = normalized_features[indices]
        shuffled_labels = encoded_labels[indices]

        train_x = shuffled_features[:size_of_train_subset]
        test_x = shuffled_features[size_of_train_subset:]

        train_y = shuffled_labels[:size_of_train_subset]
        test_y = shuffled_labels[size_of_train_subset:]

        assert len(train_x) == len(train_y)
        assert len(test_x) == len(test_y)
        assert len(train_x) == size_of_train_subset
        assert len(test_x) == size_of_test_subset

        # Check the assertions
        logging.info(
            f"Train set size: {train_x.shape}, {train_x.shape[0] / size_of_dataset:.4f} percent of the dataset"
        )
        logging.info(
            f"Test set size: {test_x.shape}, {test_x.shape[0] / size_of_dataset:.4f} percent of the dataset"
        )

        file_save_name = general_dataset_configs(dataset_name).get("cached_dataset_file")
        np.savez(file_save_name,
                 train_x=train_x,
                 test_x=test_x,
                 train_y=train_y,
                 test_y=test_y)

        logging.info(f"Saved dataset to {file_save_name}")

    except FileNotFoundError:
        logging.error("File not found")


def main():
    datasets = ["connect4", "isolete", "letter", "mnist", "mnist_fashion", "musk2", "optdigits", "page_blocks",
                "segment", "shuttle", "usps"]
    for dataset in datasets:
        split_dataset(dataset)


if __name__ == "__main__":
    main()
