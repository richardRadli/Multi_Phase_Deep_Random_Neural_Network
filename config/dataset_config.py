import os

from typing import Dict

from config.data_paths import (DATASET_FILES_PATHS, DEV_DRNN_PATHS)


def general_dataset_configs(dataset_type) -> Dict:
    dataset_config = {
        "adult": {
            "dataset_name":
                "adult",
            "dataset_size":
                32561,
            "num_train_data":
                22792,
            "num_features":
                13,
            "num_classes":
                2,
            "class_labels":
                ["<=50K", ">50K"],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_adult"), "adult.data"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_adult"), "adult.npz")
        },
        "cifar10": {
            "dataset_name":
                "cifar10",
            "dataset_size":
                60000,
            "num_train_data":
                42000,
            "num_features":
                3072,
            "num_classes":
                10,
            "class_labels":
                ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_cifar10"), "cifar10.npz"),
            "original_dataset":
                DATASET_FILES_PATHS.get_data_path("dataset_path_cifar10")
        },
        "connect4": {
            "dataset_name":
                "connect4",
            "dataset_size":
                67557,
            "num_train_data":
                47290,
            "num_features":
                42,
            "num_classes":
                3,
            "class_labels":
                ["x", "o", "b"],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_connect4"), "data.txt"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_connect4"), "connect4.npz")
        },
        "isolete": {
            "dataset_name":
                "isolete",
            "dataset_size":
                7797,
            "num_train_data":
                5458,
            "num_features":
                617,
            "num_classes":
                26,
            "class_labels":
                ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
                 "18", "19", "20", "21", "22", "23", "24", "25", "26"],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_isolete"), "data.txt"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_isolete"), "isolete.npz")
        },
        "letter": {
            "dataset_name":
                "letter",
            "dataset_size":
                20000,
            "num_train_data":
                14000,
            "num_features":
                16,
            "num_classes":
                26,
            "class_labels":
                ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_letter"), "data.txt"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_letter"), "letter.npz"),
        },
        "mnist": {
            "dataset_name":
                "mnist",
            "dataset_size":
                70000,
            "num_train_data":
                49000,
            "num_features":
                784,
            "num_classes":
                10,
            "class_labels":
                ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_mnist"), "data.txt"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_mnist"), "mnist.npz"),
            "original_dataset":
                DATASET_FILES_PATHS.get_data_path("dataset_path_mnist")
        },
        "mnist_fashion": {
            "dataset_name":
                "mnist_fashion",
            "dataset_size":
                70000,
            "num_train_data":
                49000,
            "num_features":
                784,
            "num_classes":
                10,
            "class_labels":
                ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                 "Sneaker", "Bag", "Ankle boot"],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_mnist_fashion"), "data.txt"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_mnist_fashion"), "mnist_fashion.npz"),
            "original_dataset":
                DATASET_FILES_PATHS.get_data_path("dataset_path_mnist_fashion")
        },
        "musk2": {
            "dataset_name":
                "musk2",
            "dataset_size":
                6598,
            "num_train_data":
                4619,
            "num_features":
                168,
            "num_classes":
                2,
            "class_labels":
                ["Musks", "Non musks"],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_musk2"), "data.txt"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_musk2"), "musk2.npz")
        },
        "optdigits": {
            "dataset_name":
                "optdigits",
            "dataset_size":
                5620,
            "num_train_data":
                3934,
            "num_features":
                64,
            "num_classes":
                10,
            "class_labels":
                ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_optdigits"), "data.txt"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_optdigits"), "optdigits.npz")
        },
        "page_blocks": {
            "dataset_name":
                "page_blocks",
            "dataset_size":
                5473,
            "num_train_data":
                4925,
            "num_features":
                10,
            "num_classes":
                5,
            "class_labels":
                ["text", "horiz. line", "graphic", "vert. line ", "picture"],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_page_blocks"), "data.txt"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_page_blocks"), "page_blocks.npz")
        },
        "satimages": {
            "dataset_name":
                "satimages",
            "dataset_size":
                6435,
            "num_train_data":
                4504,
            "num_features":
                36,
            "num_classes":
                6,
            "class_labels":
                [],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_satimages"), "data.txt"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_satimages"), "satimages.npz")
        },
        "segment": {
            "dataset_name":
                "segment",
            "dataset_size":
                2310,
            "num_train_data":
                1617,
            "num_features":
                19,
            "num_classes":
                7,
            "class_labels":
                ["brickface", "sky", "foliage", "cement", "window", "path", "grass"],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_segment"), "data.txt"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_segment"), "segment.npz")
        },
        "shuttle": {
            "dataset_name":
                "shuttle",
            "dataset_size":
                58000,
            "num_train_data":
                40600,
            "num_features":
                9,
            "num_classes":
                7,
            "class_labels":
                ["Rad Flow", "Fpv Close", "Fpv Open", "High", "Bypass", "Bpv Close", "Bpv Open"],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_shuttle"), "data.txt"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_shuttle"), "shuttle.npz")
        },
        "spambase": {
            "dataset_name":
                "spambase",
            "dataset_size":
                4601,
            "num_train_data":
                3220,
            "num_features":
                57,
            "num_classes":
                2,
            "class_labels":
                ["0", "1"],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_spambase"), "data.txt"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_spambase"), "spambase.npz"),
        },
        "usps": {
            "dataset_name":
                "usps",
            "dataset_size":
                9298,
            "num_train_data":
                6509,
            "num_features":
                256,
            "num_classes":
                10,
            "class_labels":
                ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_usps"), "data.txt"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_usps"), "usps.npz")
        },

        "wall": {
            "dataset_name":
                "wall",
            "dataset_size":
                5456,
            "num_train_data":
                3819,
            "num_features":
                24,
            "num_classes":
                4,
            "class_labels":
                ["0", "1", "2", "3"],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_wall"), "sensor_readings_24.data"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_wall"), "wall.npz")
        },

        "waveform": {
            "dataset_name":
                "waveform",
            "dataset_size":
                5000,
            "num_train_data":
                3500,
            "num_features":
                21,
            "num_classes":
                3,
            "class_labels":
                ["0", "1", "2"],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_waveform"), "waveform.data"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_waveform"), "waveform.npz")
        }
    }

    if dataset_type not in dataset_config:
        raise ValueError(f'Invalid dataset name: {dataset_type}')

    return dataset_config[dataset_type]


def drnn_paths_config(dataset_type) -> Dict:
    dataset_config = {
        "adult": {
            "dev_drnn": {
                "path_to_results":
                    DEV_DRNN_PATHS.get_data_path("results_adult"),
                "hyperparam_tuning":
                    DEV_DRNN_PATHS.get_data_path("hyperparam_adult"),
            }
        },
        "connect4": {
            "dev_drnn": {
                "path_to_results":
                    DEV_DRNN_PATHS.get_data_path("results_connect4"),
                "hyperparam_tuning":
                    DEV_DRNN_PATHS.get_data_path("hyperparam_connect4"),
            }
        },
        "cifar10": {
            "dev_drnn": {
                "path_to_results":
                    DEV_DRNN_PATHS.get_data_path("results_cifar10"),
                "hyperparam_tuning":
                    DEV_DRNN_PATHS.get_data_path("hyperparam_cifar10"),
            }
        },
        "isolete": {
            "dev_drnn": {
                "path_to_results":
                    DEV_DRNN_PATHS.get_data_path("results_isolete"),
                "hyperparam_tuning":
                    DEV_DRNN_PATHS.get_data_path("hyperparam_isolete"),
            }
        },
        "letter": {
            "dev_drnn": {
                "path_to_results":
                    DEV_DRNN_PATHS.get_data_path("results_letter"),
                "hyperparam_tuning":
                    DEV_DRNN_PATHS.get_data_path("hyperparam_letter"),
            },
        },
        "mnist": {
            "dev_drnn": {
                "path_to_results":
                    DEV_DRNN_PATHS.get_data_path("results_mnist"),
                "hyperparam_tuning":
                    DEV_DRNN_PATHS.get_data_path("hyperparam_mnist"),
            },
        },
        "mnist_fashion": {
            "dev_drnn": {
                "path_to_results":
                    DEV_DRNN_PATHS.get_data_path("results_mnist_fashion"),
                "hyperparam_tuning":
                    DEV_DRNN_PATHS.get_data_path("hyperparam_mnist_fashion"),
            },
        },
        "musk2": {
            "dev_drnn": {
                "path_to_results":
                    DEV_DRNN_PATHS.get_data_path("results_musk2"),
                "hyperparam_tuning":
                    DEV_DRNN_PATHS.get_data_path("hyperparam_musk2"),
            },
        },
        "optdigits": {
            "dev_drnn": {
                "path_to_results":
                    DEV_DRNN_PATHS.get_data_path("results_optdigits"),
                "hyperparam_tuning":
                    DEV_DRNN_PATHS.get_data_path("hyperparam_optdigits"),
            },
        },
        "page_blocks": {
            "dev_drnn": {
                "path_to_results":
                    DEV_DRNN_PATHS.get_data_path("results_page_blocks"),
                "hyperparam_tuning":
                    DEV_DRNN_PATHS.get_data_path("hyperparam_page_blocks"),
            },
        },
        "satimages": {
            "dev_drnn": {
                "path_to_results":
                    DEV_DRNN_PATHS.get_data_path("results_satimages"),
                "hyperparam_tuning":
                    DEV_DRNN_PATHS.get_data_path("hyperparam_satimages")
            },
        },
        "segment": {
            "dev_drnn": {
                "path_to_results":
                    DEV_DRNN_PATHS.get_data_path("results_segment"),
                "hyperparam_tuning":
                    DEV_DRNN_PATHS.get_data_path("hyperparam_segment"),
            },
        },
        "shuttle": {
            "dev_drnn": {
                "path_to_results":
                    DEV_DRNN_PATHS.get_data_path("results_shuttle"),
                "hyperparam_tuning":
                    DEV_DRNN_PATHS.get_data_path("hyperparam_shuttle"),
            },
        },
        "spambase": {
            "dev_drnn": {
                "path_to_results":
                    DEV_DRNN_PATHS.get_data_path("results_spambase"),
                "hyperparam_tuning":
                    DEV_DRNN_PATHS.get_data_path("hyperparam_spambase"),
            },
        },
        "usps": {
            "dev_drnn": {
                "path_to_results":
                    DEV_DRNN_PATHS.get_data_path("results_usps"),
                "hyperparam_tuning":
                    DEV_DRNN_PATHS.get_data_path("hyperparam_usps"),
            },
        },
        "wall": {
            "dev_drnn": {
                "path_to_results":
                    DEV_DRNN_PATHS.get_data_path("results_wall"),
                "hyperparam_tuning":
                    DEV_DRNN_PATHS.get_data_path("hyperparam_wall"),
            },
        },
        "waveform": {
            "dev_drnn": {
                "path_to_results":
                    DEV_DRNN_PATHS.get_data_path("results_waveform"),
                "hyperparam_tuning":
                    DEV_DRNN_PATHS.get_data_path("hyperparam_waveform"),
            },
        }
    }

    if dataset_type not in dataset_config:
        raise ValueError(f'Invalid dataset name: {dataset_type}')

    return dataset_config[dataset_type]

