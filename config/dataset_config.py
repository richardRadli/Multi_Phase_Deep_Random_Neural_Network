import os
from typing import Dict

from config.data_paths import (
    DATASET_FILES_PATHS,
    MPDRNN_PATHS,
    IPMPDRNN_PATHS,
    FCNN_PATHS,
    HELM_PATHS
)

VALID_DATASETS = [
    "connect4", "isolete", "letter", "mnist", "mnist_fashion",
    "musk2", "optdigits", "page_blocks", "satimages", "segment",
    "shuttle", "spambase", "usps", "wall", "waveform"
]


def _validate_dataset(dataset_type: str) -> None:
    """Helper function to prevent code repetition for validation."""
    if dataset_type not in VALID_DATASETS:
        raise ValueError(f"Invalid dataset name: {dataset_type}")


def general_dataset_configs(dataset_type: str) -> Dict:
    _validate_dataset(dataset_type)

    raw_configs = {
        "connect4": {
            "dataset_size": 67557, "num_train_data": 47290, "num_test_data": 20267,
            "num_features": 42, "num_classes": 3, "class_labels": ["x", "o", "b"]
        },
        "isolete": {
            "dataset_size": 7797, "num_train_data": 5458, "num_test_data": 2339,
            "num_features": 617, "num_classes": 26,
            "class_labels": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
                             "17", "18", "19", "20", "21", "22", "23", "24", "25", "26"]
        },
        "letter": {
            "dataset_size": 20000, "num_train_data": 14000, "num_test_data": 6000,
            "num_features": 16, "num_classes": 26,
            "class_labels": ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                             's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        },
        "mnist": {
            "dataset_size": 70000, "num_train_data": 49000, "num_test_data": 21000,
            "num_features": 784, "num_classes": 10, "class_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        },
        "mnist_fashion": {
            "dataset_size": 70000, "num_train_data": 49000, "num_test_data": 21000,
            "num_features": 784, "num_classes": 10,
            "class_labels": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                             "Ankle boot"]
        },
        "musk2": {
            "dataset_size": 6598, "num_train_data": 4619, "num_test_data": 1979,
            "num_features": 168, "num_classes": 2, "class_labels": ["Musks", "Non musks"]
        },
        "optdigits": {
            "dataset_size": 5620, "num_train_data": 3934, "num_test_data": 1686,
            "num_features": 64, "num_classes": 10, "class_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        },
        "page_blocks": {
            "dataset_size": 5473, "num_train_data": 4925, "num_test_data": 548,
            "num_features": 10, "num_classes": 5,
            "class_labels": ["text", "horiz. line", "graphic", "vert. line ", "picture"]
        },
        "satimages": {
            "dataset_size": 6435, "num_train_data": 4504, "num_test_data": 1931,
            "num_features": 36, "num_classes": 6, "class_labels": []
        },
        "segment": {
            "dataset_size": 2310, "num_train_data": 1617, "num_test_data": 693,
            "num_features": 19, "num_classes": 7,
            "class_labels": ["brickface", "sky", "foliage", "cement", "window", "path", "grass"]
        },
        "shuttle": {
            "dataset_size": 58000, "num_train_data": 40600, "num_test_data": 17400,
            "num_features": 9, "num_classes": 7,
            "class_labels": ["Rad Flow", "Fpv Close", "Fpv Open", "High", "Bypass", "Bpv Close", "Bpv Open"]
        },
        "spambase": {
            "dataset_size": 4601, "num_train_data": 3220, "num_test_data": 1381,
            "num_features": 57, "num_classes": 2, "class_labels": ["0", "1"]
        },
        "usps": {
            "dataset_size": 9298, "num_train_data": 6509, "num_test_data": 2789,
            "num_features": 256, "num_classes": 10, "class_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        },
        "wall": {
            "dataset_size": 5456, "num_train_data": 3819, "num_test_data": 1637,
            "num_features": 24, "num_classes": 4, "class_labels": ["0", "1", "2", "3"]
        },
        "waveform": {
            "dataset_size": 5000, "num_train_data": 3500, "num_test_data": 1500,
            "num_features": 21, "num_classes": 3, "class_labels": ["0", "1", "2"]
        }
    }

    config = raw_configs[dataset_type]
    base_path = DATASET_FILES_PATHS.get_data_path(f"dataset_path_{dataset_type}")

    if dataset_type == "wall":
        data_filename = "sensor_readings_24.data"
    elif dataset_type == "waveform":
        data_filename = "waveform.data"
    else:
        data_filename = "data.txt"

    config["dataset_name"] = dataset_type
    config["dataset_file"] = os.path.join(base_path, data_filename)
    config["cached_dataset_file"] = os.path.join(base_path, f"{dataset_type}.npz")

    return config


def drnn_paths_config(dataset_type: str) -> Dict:
    _validate_dataset(dataset_type)
    return {
        "mpdrnn": {
            "path_to_results":   MPDRNN_PATHS.get_data_path(f"results_{dataset_type}"),
            "hyperparam_tuning": MPDRNN_PATHS.get_data_path(f"hyperparam_{dataset_type}"),
        },
        "ipmpdrnn": {
            "path_to_results":   IPMPDRNN_PATHS.get_data_path(f"results_{dataset_type}"),
            "hyperparam_tuning": IPMPDRNN_PATHS.get_data_path(f"hyperparam_{dataset_type}"),
        }
    }


def fcnn_paths_configs(dataset_type: str) -> Dict:
    _validate_dataset(dataset_type)
    return {
        "fcnn_saved_weights": FCNN_PATHS.get_data_path(f"sw_{dataset_type}"),
        "logs":               FCNN_PATHS.get_data_path(f"logs_{dataset_type}"),
        "saved_results":      FCNN_PATHS.get_data_path(f"results_{dataset_type}"),
        "hyperparam_tuning":  FCNN_PATHS.get_data_path(f"hyperparam_tuning_{dataset_type}"),
    }


def helm_paths_config(dataset_type: str) -> Dict:
    _validate_dataset(dataset_type)
    return {
        "path_to_cm":        HELM_PATHS.get_data_path(f"cm_{dataset_type}"),
        "path_to_results":   HELM_PATHS.get_data_path(f"results_{dataset_type}"),
        "hyperparam_tuning": HELM_PATHS.get_data_path(f"hyperparam_tuning_{dataset_type}")
    }