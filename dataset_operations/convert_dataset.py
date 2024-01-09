import numpy as np

from keras.utils import np_utils
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from config.dataset_config import general_dataset_configs
from config.config import MPDRNNConfig


def main():
    # connect4, isolete
    cfg = MPDRNNConfig().parse()

    path_to_dataset = general_dataset_configs(cfg).get("dataset_file")

    with open(path_to_dataset, "r") as file:
        lines = file.readlines()

    # Split the data into labels and features
    labels = []
    features = []
    for line in tqdm(lines):
        split = line.strip().split(',')
        labels.append(split[-1])
        features.append(split[:-1])

    # Convert the features to numerical values using label encoding
    encoder = LabelEncoder()
    encoded_features = []
    for feature in tqdm(features):
        encoded_features.append(encoder.fit_transform(feature))

    # One hot encode the labels
    encoded_labels = encoder.fit_transform(labels)
    one_hot_labels = np_utils.to_categorical(encoded_labels)

    train_x, test_x, train_y, test_y = train_test_split(encoded_features,
                                                        one_hot_labels,
                                                        test_size=general_dataset_configs(cfg).get("num_test_data"),
                                                        random_state=42)

    scaler = MinMaxScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    file_save_name = general_dataset_configs(cfg).get("cached_dataset_file")
    np.save(str(file_save_name), [train_x_scaled, test_x_scaled, train_y, test_y])


if __name__ == "__main__":
    main()
