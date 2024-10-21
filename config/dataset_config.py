import os

from typing import Dict

from config.data_paths import DATASET_FILES_PATHS, MPDRNN_PATHS, IPMPDRNN_PATHS, FCNN_PATHS, HELM_PATHS


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
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_mnist"), "mnist.npz")
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
            "mpdrnn": {
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_adult"),
                "hyperparam_tuning":
                    MPDRNN_PATHS.get_data_path("hyperparam_adult"),
            },
            "ipmpdrnn": {
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_adult"),
                "hyperparam_tuning":
                    IPMPDRNN_PATHS.get_data_path("hyperparam_adult"),
            }
        },
        "connect4": {
            "mpdrnn": {
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_connect4"),
                "hyperparam_tuning":
                    MPDRNN_PATHS.get_data_path("hyperparam_connect4"),
            },
            "ipmpdrnn": {
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_connect4"),
                "hyperparam_tuning":
                    IPMPDRNN_PATHS.get_data_path("hyperparam_connect4"),
            }
        },
        "isolete": {
            "mpdrnn": {
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_isolete"),
                "hyperparam_tuning":
                    MPDRNN_PATHS.get_data_path("hyperparam_isolete"),
            },
            "ipmpdrnn": {
                "path_to_cm":
                    IPMPDRNN_PATHS.get_data_path("cm_isolete"),
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_isolete"),
                "hyperparam_tuning":
                    IPMPDRNN_PATHS.get_data_path("hyperparam_isolete"),
            }
        },
        "letter": {
            "mpdrnn": {
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_letter"),
                "hyperparam_tuning":
                    MPDRNN_PATHS.get_data_path("hyperparam_letter"),
            },
            "ipmpdrnn": {
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_letter"),
                "hyperparam_tuning":
                    IPMPDRNN_PATHS.get_data_path("hyperparam_letter"),
            }
        },
        "mnist": {
            "mpdrnn": {
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_mnist"),
                "hyperparam_tuning":
                    MPDRNN_PATHS.get_data_path("hyperparam_mnist"),
            },
            "ipmpdrnn": {
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_mnist"),
                "hyperparam_tuning":
                    IPMPDRNN_PATHS.get_data_path("hyperparam_mnist"),
            }
        },
        "mnist_fashion": {
            "mpdrnn": {
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_mnist_fashion"),
                "hyperparam_tuning":
                    MPDRNN_PATHS.get_data_path("hyperparam_mnist_fashion"),
            },
            "ipmpdrnn": {
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_mnist_fashion"),
                "hyperparam_tuning":
                    IPMPDRNN_PATHS.get_data_path("hyperparam_mnist_fashion"),
            }
        },
        "musk2": {
            "mpdrnn": {
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_musk2"),
                "hyperparam_tuning":
                    MPDRNN_PATHS.get_data_path("hyperparam_musk2"),
            },
            "ipmpdrnn": {
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_musk2"),
                "hyperparam_tuning":
                    IPMPDRNN_PATHS.get_data_path("hyperparam_musk2"),
            }
        },
        "optdigits": {
            "mpdrnn": {
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_optdigits"),
                "hyperparam_tuning":
                    MPDRNN_PATHS.get_data_path("hyperparam_optdigits"),
            },
            "ipmpdrnn": {
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_optdigits"),
                "hyperparam_tuning":
                    IPMPDRNN_PATHS.get_data_path("hyperparam_optdigits"),
            }
        },
        "page_blocks": {
            "mpdrnn": {
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_page_blocks"),
                "hyperparam_tuning":
                    MPDRNN_PATHS.get_data_path("hyperparam_page_blocks"),
            },
            "ipmpdrnn": {
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_page_blocks"),
                "hyperparam_tuning":
                    IPMPDRNN_PATHS.get_data_path("hyperparam_page_blocks"),
            }
        },
        "satimages": {
            "mpdrnn": {
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_satimages"),
                "hyperparam_tuning":
                    MPDRNN_PATHS.get_data_path("hyperparam_satimages")
            },
            "ipmpdrnn": {
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_satimages"),
                "hyperparam_tuning":
                    IPMPDRNN_PATHS.get_data_path("hyperparam_satimages")
            }
        },
        "segment": {
            "mpdrnn": {
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_segment"),
                "hyperparam_tuning":
                    MPDRNN_PATHS.get_data_path("hyperparam_segment"),
            },
            "ipmpdrnn": {
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_segment"),
                "hyperparam_tuning":
                    IPMPDRNN_PATHS.get_data_path("hyperparam_segment")
            }
        },
        "shuttle": {
            "mpdrnn": {
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_shuttle"),
                "hyperparam_tuning":
                    MPDRNN_PATHS.get_data_path("hyperparam_shuttle"),
            },
            "ipmpdrnn": {
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_shuttle"),
                "hyperparam_tuning":
                    IPMPDRNN_PATHS.get_data_path("hyperparam_shuttle"),
            }
        },
        "spambase": {
            "mpdrnn": {
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_spambase"),
                "hyperparam_tuning":
                    MPDRNN_PATHS.get_data_path("hyperparam_spambase"),
            },
            "ipmpdrnn": {
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_spambase"),
                "hyperparam_tuning":
                    IPMPDRNN_PATHS.get_data_path("hyperparam_spambase"),
            }
        },
        "usps": {
            "mpdrnn": {
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_usps"),
                "hyperparam_tuning":
                    MPDRNN_PATHS.get_data_path("hyperparam_usps"),
            },
            "ipmpdrnn": {
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_usps"),
                "hyperparam_tuning":
                    IPMPDRNN_PATHS.get_data_path("hyperparam_usps"),
            }
        },
        "wall": {
            "mpdrnn": {
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_wall"),
                "hyperparam_tuning":
                    MPDRNN_PATHS.get_data_path("hyperparam_wall"),
            },
            "ipmpdrnn": {
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_wall"),
                "hyperparam_tuning":
                    IPMPDRNN_PATHS.get_data_path("hyperparam_wall"),
            }
        },
        "waveform": {
            "mpdrnn": {
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_waveform"),
                "hyperparam_tuning":
                    MPDRNN_PATHS.get_data_path("hyperparam_waveform"),
            },
            "ipmpdrnn": {
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_waveform"),
                "hyperparam_tuning":
                    IPMPDRNN_PATHS.get_data_path("hyperparam_waveform"),
            }
        }
    }

    if dataset_type not in dataset_config:
        raise ValueError(f'Invalid dataset name: {dataset_type}')

    return dataset_config[dataset_type]


def fcnn_paths_configs(dataset_type) -> Dict:
    dataset_config = {
        "adult": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_adult"),
            "logs":
                FCNN_PATHS.get_data_path("logs_adult"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_adult"),
            "hyperparam_tuning":
                FCNN_PATHS.get_data_path("hyperparam_tuning_adult"),
        },
        "connect4": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_connect4"),
            "logs":
                FCNN_PATHS.get_data_path("logs_connect4"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_connect4"),
            "hyperparam_tuning":
                FCNN_PATHS.get_data_path("hyperparam_tuning_connect4"),
        },
        "isolete": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_isolete"),
            "logs":
                FCNN_PATHS.get_data_path("logs_isolete"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_isolete"),
            "hyperparam_tuning":
                FCNN_PATHS.get_data_path("hyperparam_tuning_isolete"),
        },
        "letter": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_letter"),
            "logs":
                FCNN_PATHS.get_data_path("logs_letter"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_letter"),
            "hyperparam_tuning":
                FCNN_PATHS.get_data_path("hyperparam_tuning_letter"),
        },
        "mnist": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_mnist"),
            "logs":
                FCNN_PATHS.get_data_path("logs_mnist"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_mnist"),
            "hyperparam_tuning":
                FCNN_PATHS.get_data_path("hyperparam_tuning_mnist"),
        },
        "mnist_fashion": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_mnist_fashion"),
            "logs":
                FCNN_PATHS.get_data_path("logs_mnist_fashion"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_mnist_fashion"),
            "hyperparam_tuning":
                FCNN_PATHS.get_data_path("hyperparam_tuning_mnist_fashion"),
        },
        "musk2": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_musk2"),
            "logs":
                FCNN_PATHS.get_data_path("logs_musk2"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_musk2"),
            "hyperparam_tuning":
                FCNN_PATHS.get_data_path("hyperparam_tuning_musk2"),
        },
        "optdigits": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_optdigits"),
            "logs":
                FCNN_PATHS.get_data_path("logs_optdigits"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_optdigits"),
            "hyperparam_tuning":
                FCNN_PATHS.get_data_path("hyperparam_tuning_optdigits"),
        },
        "page_blocks": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_page_blocks"),
            "logs":
                FCNN_PATHS.get_data_path("logs_page_blocks"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_page_blocks"),
            "hyperparam_tuning":
                FCNN_PATHS.get_data_path("hyperparam_tuning_page_blocks"),
        },
        "satimages": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_satimages"),
            "logs":
                FCNN_PATHS.get_data_path("logs_satimages"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_satimages"),
            "hyperparam_tuning":
                FCNN_PATHS.get_data_path("hyperparam_tuning_satimages"),
        },
        "segment": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_segment"),
            "logs":
                FCNN_PATHS.get_data_path("logs_segment"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_segment"),
            "hyperparam_tuning":
                FCNN_PATHS.get_data_path("hyperparam_tuning_segment"),
        },
        "shuttle": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_shuttle"),
            "logs":
                FCNN_PATHS.get_data_path("logs_shuttle"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_shuttle"),
            "hyperparam_tuning":
                FCNN_PATHS.get_data_path("hyperparam_tuning_shuttle"),
        },
        "spambase": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_spambase"),
            "logs":
                FCNN_PATHS.get_data_path("logs_spambase"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_spambase"),
            "hyperparam_tuning":
                FCNN_PATHS.get_data_path("hyperparam_tuning_spambase"),
        },
        "usps": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_usps"),
            "logs":
                FCNN_PATHS.get_data_path("logs_usps"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_usps"),
            "hyperparam_tuning":
                FCNN_PATHS.get_data_path("hyperparam_tuning_usps"),
        },
        "wall": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_wall"),
            "logs":
                FCNN_PATHS.get_data_path("logs_wall"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_wall"),
            "hyperparam_tuning":
                FCNN_PATHS.get_data_path("hyperparam_tuning_wall"),
        },
        "waveform": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_waveform"),
            "logs":
                FCNN_PATHS.get_data_path("logs_waveform"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_waveform"),
            "hyperparam_tuning":
                FCNN_PATHS.get_data_path("hyperparam_tuning_waveform"),
        }
    }

    if dataset_type not in dataset_config:
        raise ValueError(f"Invalid dataset name: {dataset_type}")

    return dataset_config[dataset_type]


def helm_paths_config(dataset_type) -> Dict:
    dataset_config = {
        "adult": {
            "path_to_cm":
                HELM_PATHS.get_data_path("cm_adult"),
            "path_to_results":
                HELM_PATHS.get_data_path("results_adult"),
            "hyperparam_tuning":
                HELM_PATHS.get_data_path("hyperparam_tuning_adult")
        },
        "connect4": {
            "path_to_cm":
                HELM_PATHS.get_data_path("cm_connect4"),
            "path_to_results":
                HELM_PATHS.get_data_path("results_connect4"),
            "hyperparam_tuning":
                HELM_PATHS.get_data_path("hyperparam_tuning_connect4")
        },
        "isolete": {
            "path_to_cm":
                HELM_PATHS.get_data_path("cm_isolete"),
            "path_to_results":
                HELM_PATHS.get_data_path("results_isolete"),
            "hyperparam_tuning":
                HELM_PATHS.get_data_path("hyperparam_tuning_isolete")
        },
        "letter": {
            "path_to_cm":
                HELM_PATHS.get_data_path("cm_letter"),
            "path_to_results":
                HELM_PATHS.get_data_path("results_letter"),
            "hyperparam_tuning":
                HELM_PATHS.get_data_path("hyperparam_tuning_letter")
        },
        "mnist": {
            "path_to_cm":
                HELM_PATHS.get_data_path("cm_mnist"),
            "path_to_results":
                HELM_PATHS.get_data_path("results_mnist"),
            "hyperparam_tuning":
                HELM_PATHS.get_data_path("hyperparam_tuning_mnist")
        },
        "mnist_fashion": {
            "path_to_cm":
                HELM_PATHS.get_data_path("cm_mnist_fashion"),
            "path_to_results":
                HELM_PATHS.get_data_path("results_mnist_fashion"),
            "hyperparam_tuning":
                HELM_PATHS.get_data_path("hyperparam_tuning_mnist_fashion")
        },
        "musk2": {
            "path_to_cm":
                HELM_PATHS.get_data_path("cm_musk2"),
            "path_to_results":
                HELM_PATHS.get_data_path("results_musk2"),
            "hyperparam_tuning":
                HELM_PATHS.get_data_path("hyperparam_tuning_musk2")
        },
        "optdigits": {
            "path_to_cm":
                HELM_PATHS.get_data_path("cm_optdigits"),
            "path_to_results":
                HELM_PATHS.get_data_path("results_optdigits"),
            "hyperparam_tuning":
                HELM_PATHS.get_data_path("hyperparam_tuning_optdigits")
        },
        "page_blocks": {
            "path_to_cm":
                HELM_PATHS.get_data_path("cm_page_blocks"),
            "path_to_results":
                HELM_PATHS.get_data_path("results_page_blocks"),
            "hyperparam_tuning":
                HELM_PATHS.get_data_path("hyperparam_tuning_page_blocks")
        },
        "satimages": {
            "path_to_cm":
                HELM_PATHS.get_data_path("cm_satimages"),
            "path_to_results":
                HELM_PATHS.get_data_path("results_satimages"),
            "hyperparam_tuning":
                HELM_PATHS.get_data_path("hyperparam_tuning_satimages")
        },
        "segment": {
            "path_to_cm":
                HELM_PATHS.get_data_path("cm_segment"),
            "path_to_results":
                HELM_PATHS.get_data_path("results_segment"),
            "hyperparam_tuning":
                HELM_PATHS.get_data_path("hyperparam_tuning_segment")
        },
        "shuttle": {
            "path_to_cm":
                HELM_PATHS.get_data_path("cm_shuttle"),
            "path_to_results":
                HELM_PATHS.get_data_path("results_shuttle"),
            "hyperparam_tuning":
                HELM_PATHS.get_data_path("hyperparam_tuning_shuttle")
        },
        "spambase": {
            "path_to_cm":
                HELM_PATHS.get_data_path("cm_spambase"),
            "path_to_results":
                HELM_PATHS.get_data_path("results_spambase"),
            "hyperparam_tuning":
                HELM_PATHS.get_data_path("hyperparam_tuning_spambase")
        },
        "usps": {
            "path_to_cm":
                HELM_PATHS.get_data_path("cm_usps"),
            "path_to_results":
                HELM_PATHS.get_data_path("results_usps"),
            "hyperparam_tuning":
                HELM_PATHS.get_data_path("hyperparam_tuning_usps")
        },
        "wall": {
            "path_to_cm":
                HELM_PATHS.get_data_path("cm_wall"),
            "path_to_results":
                HELM_PATHS.get_data_path("results_wall"),
            "hyperparam_tuning":
                HELM_PATHS.get_data_path("hyperparam_tuning_wall")
        },
        "waveform": {
            "path_to_cm":
                HELM_PATHS.get_data_path("cm_waveform"),
            "path_to_results":
                HELM_PATHS.get_data_path("results_waveform"),
            "hyperparam_tuning":
                HELM_PATHS.get_data_path("hyperparam_tuning_waveform")
        }
    }

    if dataset_type not in dataset_config:
        raise ValueError(f'Invalid dataset name: {dataset_type}')

    return dataset_config[dataset_type]
