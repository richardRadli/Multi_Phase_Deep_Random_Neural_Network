import logging

from typing import Dict

from elm.src.config.const import DATASET_FCNN_PATHS, DATASET_FILES_PATHS


def general_dataset_configs(cfg) -> Dict:
    dataset_type = cfg.dataset_name
    dataset_config = {
        "connect4": {
            "dataset_name":
                "connect4",
            "num_train_data":
                49980,
            "num_test_data":
                17577,
            "num_features":
                42,
            "num_classes":
                3,
            "eq_neurons":
                [866, 866, 866],
            "exp_neurons":
                [2049, 452, 100],
            "helm_neurons":
                [400, 200, 2000],
            "class_labels":
                ["x", "o", "b"],
            "cached_dataset_file":
                DATASET_FILES_PATHS.get_data_path("dataset_path_connect4")
        },
        "isolete": {
            "dataset_name":
                "isolete",
            "num_train_data":
                6238,
            "num_test_data":
                1559,
            "num_features":
                617,
            "num_classes":
                27,
            "eq_neurons":
                [1300, 1300, 1300],
            "exp_neurons":
                [2000, 1500, 600],
            "helm_neurons":
                [800, 400, 3000],
            "class_labels":
                [],
            "cached_dataset_file":
                DATASET_FILES_PATHS.get_data_path("dataset_path_isolete")
        },
        "letter": {
            "dataset_name":
                "letter",
            "num_train_data":
                10500,
            "num_test_data":
                9500,
            "num_features":
                16,
            "num_classes":
                26,
            "eq_neurons":
                [216, 216, 216],
            "exp_neurons":
                [400, 200, 50],
            "helm_neurons":
                [100, 50, 500],
            "class_labels":
                ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
            "cached_dataset_file":
                DATASET_FILES_PATHS.get_data_path("dataset_path_letter")
        },
        "mnist": {
            "dataset_name":
                "mnist",
            "num_train_data":
                60000,
            "num_test_data":
                10000,
            "num_features":
                784,
            "num_classes":
                10,
            "eq_neurons":
                [5333, 5333, 5333],
            "exp_neurons":
                [8000, 5000, 3000],
            "helm_neurons":
                [1000, 5000, 10000],
            "class_labels":
                ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "cached_dataset_file":
                DATASET_FILES_PATHS.get_data_path("dataset_path_mnist")
        },
        "mnist_fashion": {
            "dataset_name":
                "mnist_fashion",
            "num_train_data":
                60000,
            "num_test_data":
                10000,
            "num_features":
                784,
            "num_classes":
                10,
            "eq_neurons":
                [5333, 5333, 5333],
            "exp_neurons":
                [8000, 5000, 3000],
            "helm_neurons":
                [1000, 5000, 10000],
            "class_labels":
                ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                 "Sneaker", "Bag", "Ankle boot"],
            "cached_dataset_file":
                DATASET_FILES_PATHS.get_data_path("dataset_path_mnist_fashion")
        },
        "musk2": {
            "dataset_name":
                "musk2",
            "num_train_data":
                3000,
            "num_test_data":
                3598,
            "num_features":
                168,
            "num_classes":
                2,
            "eq_neurons":
                [866, 866, 866],
            "exp_neurons":
                [1500, 750, 350],
            "helm_neurons":
                [400, 200, 2000],
            "class_labels":
                ["Musks", "Non musks"],
            "cached_dataset_file":
                DATASET_FILES_PATHS.get_data_path("dataset_path_musk2")
        },
        "optdigits": {
            "dataset_name":
                "optdigits",
            "num_train_data":
                3823,
            "num_test_data":
                1797,
            "num_features":
                64,
            "num_classes":
                10,
            "eq_neurons":
                [216, 216, 216],
            "exp_neurons":
                [325, 325],
            "helm_neurons":
                [100, 50, 500],
            "class_labels":
                ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "cached_dataset_file":
                DATASET_FILES_PATHS.get_data_path("dataset_path_optdigits")
        },
        "page_blocks": {
            "dataset_name":
                "page_blocks",
            "num_train_data":
                4372,
            "num_test_data":
                1100,
            "num_features":
                10,
            "num_classes":
                6,
            "eq_neurons":
                [216, 216, 216],
            "exp_neurons":
                [400, 200, 50],
            "helm_neurons":
                [100, 50, 500],
            "class_labels":
                ["text", "horiz. line", "graphic", "vert. line ", "picture"],
            "cached_dataset_file":
                DATASET_FILES_PATHS.get_data_path("dataset_path_page_blocks")
        },
        "segment": {
            "dataset_name":
                "segment",
            "num_train_data":
                1732,
            "num_test_data":
                577,
            "num_features":
                19,
            "num_classes":
                7,
            "eq_neurons":
                [216, 216, 216],
            "exp_neurons":
                [400, 200, 50],
            "helm_neurons":
                [100, 50, 500],
            "class_labels":
                ["brickface", "sky", "foliage", "cement", "window", "path", "grass"],
            "cached_dataset_file":
                DATASET_FILES_PATHS.get_data_path("dataset_path_segment")
        },
        "shuttle": {
            "dataset_name":
                "shuttle",
            "num_train_data":
                29834,
            "num_test_data":
                26936,
            "num_features":
                9,
            "num_classes":
                8,
            "eq_neurons":
                [216, 216, 216],
            "exp_neurons":
                [400, 200, 50],
            "helm_neurons":
                [100, 50, 500],
            "class_labels":
                ["Rad Flow", "Fpv Close", "Fpv Open", "High", "Bypass", "Bpv Close", "Bpv Open"],
            "cached_dataset_file":
                DATASET_FILES_PATHS.get_data_path("dataset_path_shuttle")
        },
        "spambase": {
            "dataset_name":
                "spambase",
            "num_train_data":
                3681,
            "num_test_data":
                920,
            "num_features":
                57,
            "num_classes":
                2,
            "eq_neurons":
                [216, 216, 216],
            "exp_neurons":
                [400, 200, 50],
            "helm_neurons":
                [100, 50, 500],
            "class_labels":
                ["0", "1"],
            "cached_dataset_file":
                DATASET_FILES_PATHS.get_data_path("dataset_path_spambase")
        },
        "usps": {
            "dataset_name":
                "usps",
            "num_train_data":
                7291,
            "num_test_data":
                2007,
            "num_features":
                256,
            "num_classes":
                10,
            "eq_neurons":
                [866, 866, 866],
            "exp_neurons":
                [2049, 452, 100],
            "helm_neurons":
                [400, 200, 2000],
            "class_labels":
                ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "cached_dataset_file":
                DATASET_FILES_PATHS.get_data_path("dataset_path_usps")
        },
        "yaleb": {
            "dataset_name":
                "yaleB",
            "num_train_data":
                1680,
            "num_test_data":
                734,
            "num_features":
                1024,
            "num_classes":
                40,
            "eq_neurons":
                [1400, 1400, 1400],
            "exp_neurons":
                [1050, 1050, 1050, 1050],
            "helm_neurons":
                [800, 400, 3000],
            "class_labels":
                [],
            "cached_dataset_file":
                DATASET_FILES_PATHS.get_data_path("dataset_path_yaleb")
        }
    }

    if dataset_type not in dataset_config:
        raise ValueError(f'Invalid dataset name: {dataset_type}')

    return dataset_config[dataset_type]


def fcnn_dataset_configs(cfg) -> Dict:
    dataset_type = cfg.dataset_name
    logging.info(dataset_type)

    dataset_config = {
        "connect4": {
            "fcnn_saved_weights":
                DATASET_FCNN_PATHS.get_data_path("sw_connect4"),
            "logs":
                DATASET_FCNN_PATHS.get_data_path("logs_connect4")
        },
        "isolete": {
            "fcnn_saved_weights":
                DATASET_FCNN_PATHS.get_data_path("sw_isolete"),
            "logs":
                DATASET_FCNN_PATHS.get_data_path("logs_isolete")
        },
        "letter": {
            "fcnn_saved_weights":
                DATASET_FCNN_PATHS.get_data_path("sw_letter"),
            "logs":
                DATASET_FCNN_PATHS.get_data_path("logs_letter")
        },
        "mnist": {
            "fcnn_saved_weights":
                DATASET_FCNN_PATHS.get_data_path("sw_mnist"),
            "logs":
                DATASET_FCNN_PATHS.get_data_path("logs_mnist")
        },
        "mnist_fashion": {
            "fcnn_saved_weights":
                DATASET_FCNN_PATHS.get_data_path("sw_mnist_fashion"),
            "logs":
                DATASET_FCNN_PATHS.get_data_path("logs_mnist_fashion")
        },
        "musk2": {
            "fcnn_saved_weights":
                DATASET_FCNN_PATHS.get_data_path("sw_musk2"),
            "logs":
                DATASET_FCNN_PATHS.get_data_path("logs_musk2")
        },
        "optdigits": {
            "fcnn_saved_weights":
                DATASET_FCNN_PATHS.get_data_path("sw_optdigits"),
            "logs":
                DATASET_FCNN_PATHS.get_data_path("logs_optdigits")
        },
        "page_blocks": {
            "fcnn_saved_weights":
                DATASET_FCNN_PATHS.get_data_path("sw_page_blocks"),
            "logs":
                DATASET_FCNN_PATHS.get_data_path("logs_page_blocks")
        },
        "segment": {
            "fcnn_saved_weights":
                DATASET_FCNN_PATHS.get_data_path("sw_segment"),
            "logs":
                DATASET_FCNN_PATHS.get_data_path("logs_segment")
        },
        "shuttle": {
            "fcnn_saved_weights":
                DATASET_FCNN_PATHS.get_data_path("sw_shuttle"),
            "logs":
                DATASET_FCNN_PATHS.get_data_path("logs_shuttle")
        },
        "spambase": {
            "fcnn_saved_weights":
                DATASET_FCNN_PATHS.get_data_path("sw_spambase"),
            "logs":
                DATASET_FCNN_PATHS.get_data_path("logs_spambase")
        },
        "usps": {
            "fcnn_saved_weights":
                DATASET_FCNN_PATHS.get_data_path("sw_usps"),
            "logs":
                DATASET_FCNN_PATHS.get_data_path("logs_usps")
        },
        "yaleb": {
            "fcnn_saved_weights":
                DATASET_FCNN_PATHS.get_data_path("sw_yaleb"),
            "logs":
                DATASET_FCNN_PATHS.get_data_path("logs_yaleb")
        }
    }

    if dataset_type not in dataset_config:
        raise ValueError(f"Invalid dataset name: {dataset_type}")

    return dataset_config[dataset_type]
