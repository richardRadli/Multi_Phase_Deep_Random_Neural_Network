import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

from config.dataset_config import general_dataset_configs
from config.config import MPDRNNConfig


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


def main():
    cfg = MPDRNNConfig().parse()

    path_to_dataset = general_dataset_configs(cfg).get("dataset_file")
    num_data = general_dataset_configs(cfg).get("num_train_data") + general_dataset_configs(cfg).get("num_test_data")
    num_features = general_dataset_configs(cfg).get("num_features")

    try:
        with open(path_to_dataset, "r") as file:
            lines = file.readlines()

        # Split the data into labels and features
        labels = []
        features = []
        for line in tqdm(lines):
            split = line.strip().split(',')
            # label at the end
            if cfg.dataset_name in ["connect4", "isolete", "iris", "musk2", "optdigits", "page_blocks", "satimages",
                                    "shuttle", "spambase", "forest", "usps"]:
                labels.append(split[-1])
                features.append(split[:-1])
            # label at the front
            elif cfg.dataset_name in ["letter", "mnist", "mnist_fashion", "segment"]:
                labels.append(split[0])
                features.append(split[1:])
            else:
                raise ValueError(f"Wrong dataset name {cfg.dataset_name}")

        label_encoder = OneHotEncoder(sparse_output=False)
        encoded_labels = label_encoder.fit_transform(np.array(labels).reshape(-1, 1))

        if all_elements_numeric(features):
            reshaped_features = np.array(features, dtype=np.float32)
            reshaped_features = reshaped_features.reshape((num_data, num_features))
        else:
            encoder = LabelEncoder()
            reshaped_features = []
            for feature in tqdm(features):
                reshaped_features.append(encoder.fit_transform(feature))

        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(reshaped_features)

        train_x, test_x, train_y, test_y = (
            train_test_split(
                normalized_features,
                encoded_labels,
                test_size=general_dataset_configs(cfg).get("num_test_data"),
                random_state=42
            )
        )

        file_save_name = general_dataset_configs(cfg).get("cached_dataset_file")
        np.save(str(file_save_name), [train_x, test_x, train_y, test_y])
    except FileNotFoundError:
        print("File not found")


if __name__ == "__main__":
    main()
