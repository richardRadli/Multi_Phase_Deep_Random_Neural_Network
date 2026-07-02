import os

from typing import Dict

from config.data_paths import DATASET_FILES_PATHS, MPDRNN_PATHS, FCNN_PATHS, HELM_PATHS

VALID_DATASETS = [
    "connect4", "isolete", "letter", "mnist", "mnist_fashion",
    "musk2", "optdigits", "page_blocks", "segment", "shuttle", "usps"
]

def _validate_dataset(dataset_type: str) -> None:
    """Helper function to prevent code repetition for validation."""
    if dataset_type not in VALID_DATASETS:
        raise ValueError(f"Invalid dataset name: {dataset_type}")


def general_dataset_configs(dataset_type) -> Dict:
    _validate_dataset(dataset_type)

    raw_configs = {
        "connect4": {
            "dataset_size": 67557, "num_train_data": 50000, "num_test_data": 17557,
            "num_features": 42, "num_classes": 3, "class_labels": ["x", "o", "b"]
        },
        "isolete": {
            "dataset_size": 7797, "num_train_data": 6238, "num_test_data": 1559,
            "num_features": 617, "num_classes": 26,
            "class_labels": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
                             "17", "18", "19", "20", "21", "22", "23", "24", "25", "26"]
        },
        "letter": {
            "dataset_size": 20000, "num_train_data": 10500, "num_test_data": 9500,
            "num_features": 16, "num_classes": 26,
            "class_labels": ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                             's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        },
        "mnist": {
            "dataset_size": 70000, "num_train_data": 60000, "num_test_data": 10000,
            "num_features": 784, "num_classes": 10, "class_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        },
        "mnist_fashion": {
            "dataset_size": 70000, "num_train_data": 60000, "num_test_data": 10000,
            "num_features": 784, "num_classes": 10,
            "class_labels": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                             "Ankle boot"]
        },
        "musk2": {
            "dataset_size": 6598, "num_train_data": 3000, "num_test_data": 3598,
            "num_features": 168, "num_classes": 2, "class_labels": ["Musks", "Non musks"]
        },
        "optdigits": {
            "dataset_size": 5620, "num_train_data": 3822, "num_test_data": 1798,
            "num_features": 64, "num_classes": 10, "class_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        },
        "page_blocks": {
            "dataset_size": 5473, "num_train_data": 4373, "num_test_data": 1100,
            "num_features": 10, "num_classes": 5,
            "class_labels": ["text", "horiz. line", "graphic", "vert. line ", "picture"]
        },
        "segment": {
            "dataset_size": 2310, "num_train_data": 1733, "num_test_data": 577,
            "num_features": 19, "num_classes": 7,
            "class_labels": ["brickface", "sky", "foliage", "cement", "window", "path", "grass"]
        },
        "shuttle": {
            "dataset_size": 58000, "num_train_data": 29834, "num_test_data": 28166,
            "num_features": 9, "num_classes": 7,
            "class_labels": ["Rad Flow", "Fpv Close", "Fpv Open", "High", "Bypass", "Bpv Close", "Bpv Open"]
        },
        "usps": {
            "dataset_size": 9298, "num_train_data": 7291, "num_test_data": 2007,
            "num_features": 256, "num_classes": 10, "class_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        }
    }

    config = raw_configs[dataset_type]

    base_path = DATASET_FILES_PATHS.get_data_path(f"dataset_path_{dataset_type}")

    config["dataset_name"] = dataset_type
    config["dataset_file"] = os.path.join(base_path, "data.txt")
    config["cached_dataset_file"] = os.path.join(base_path, f"{dataset_type}.npz")

    return config


def drnn_paths_config(dataset_type) -> Dict:
    _validate_dataset(dataset_type)
    return {
        "mpdrnn": {
            "path_to_results":    MPDRNN_PATHS.get_data_path(f"results_{dataset_type}"),
            "hyperparam_tuning":  MPDRNN_PATHS.get_data_path(f"hyperparam_{dataset_type}"),
        }
    }


def fcnn_paths_configs(dataset_type) -> Dict:
    _validate_dataset(dataset_type)
    return {
        "fcnn_saved_weights": FCNN_PATHS.get_data_path(f"sw_{dataset_type}"),
        "logs":               FCNN_PATHS.get_data_path(f"logs_{dataset_type}"),
        "saved_results":      FCNN_PATHS.get_data_path(f"results_{dataset_type}"),
        "hyperparam_tuning":  FCNN_PATHS.get_data_path(f"hyperparam_tuning_{dataset_type}"),
    }


def helm_paths_config(dataset_type) -> Dict:
    _validate_dataset(dataset_type)
    return {
        "path_to_cm":         HELM_PATHS.get_data_path(f"cm_{dataset_type}"),
        "path_to_results":    HELM_PATHS.get_data_path(f"results_{dataset_type}"),
        "hyperparam_tuning":  HELM_PATHS.get_data_path(f"hyperparam_tuning_{dataset_type}")
    }