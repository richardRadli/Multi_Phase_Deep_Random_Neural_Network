import os

from typing import Dict

from config.data_paths import DATASET_FILES_PATHS, MPDRNN_PATHS, IPMPDRNN_PATHS, FCNN_PATHS


def general_dataset_configs(dataset_type) -> Dict:
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
            "num_train_data":
                6238,
            "num_test_data":
                1559,
            "num_features":
                617,
            "num_classes":
                26,
            "eq_neurons":
                [1300, 1300, 1300],
            "exp_neurons":
                [2000, 1500, 600],
            "class_labels":
                [],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_isolete"), "data.txt"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_isolete"), "isolete.npz")
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
            "num_train_data":
                60000,
            "num_test_data":
                10000,
            "num_features":
                784,
            "num_classes":
                10,
            "eq_neurons":
                [1000, 1000, 1000],
            "exp_neurons":
                [8000, 5000, 3000],
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
                [325, 325, 325],
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
            "num_train_data":
                4373,
            "num_test_data":
                1100,
            "num_features":
                10,
            "num_classes":
                5,
            "eq_neurons":
                [216, 216, 216],
            "exp_neurons":
                [400, 200, 50],
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
            "num_train_data":
                4435,
            "num_test_data":
                2000,
            "num_features":
                36,
            "num_classes":
                6,
            "eq_neurons":
                [1000, 1000, 1000],
            "exp_neurons":
                [100, 40, 10],
            "class_labels":
                [],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_satimages"), "data.txt"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_satimages"), "satimages.npz"),
            "mpdrnn": {
                "path_to_cm":
                    MPDRNN_PATHS.get_data_path("cm_satimages"),
                "path_to_metrics":
                    MPDRNN_PATHS.get_data_path("metrics_satimages"),
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_satimages")
            },
            "ipmpdrnn": {
                "path_to_cm":
                    IPMPDRNN_PATHS.get_data_path("cm_satimages"),
                "path_to_metrics":
                    IPMPDRNN_PATHS.get_data_path("metrics_satimages"),
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_satimages")
            }
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
            "num_train_data":
                40600,
            "num_test_data":
                17400,
            "num_features":
                9,
            "num_classes":
                7,
            "eq_neurons":
                [216, 216, 216],
            "exp_neurons":
                [400, 200, 50],
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
            "class_labels":
                ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_usps"), "data.txt"),
            "cached_dataset_file":
                os.path.join(DATASET_FILES_PATHS.get_data_path("dataset_path_usps"), "usps.npz")
        }
    }

    if dataset_type not in dataset_config:
        raise ValueError(f'Invalid dataset name: {dataset_type}')

    return dataset_config[dataset_type]


def drnn_paths_config(dataset_type) -> Dict:
    dataset_config = {
        "connect4": {
            "mpdrnn": {
                "path_to_cm":
                    MPDRNN_PATHS.get_data_path("cm_connect4"),
                "path_to_metrics":
                    MPDRNN_PATHS.get_data_path("metrics_connect4"),
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_connect4")
            },
            "ipmpdrnn": {
                "path_to_cm":
                    IPMPDRNN_PATHS.get_data_path("cm_connect4"),
                "path_to_metrics":
                    IPMPDRNN_PATHS.get_data_path("metrics_connect4"),
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_connect4")
            }
        },
        "isolete": {
            "mpdrnn": {
                "path_to_cm":
                    MPDRNN_PATHS.get_data_path("cm_isolete"),
                "path_to_metrics":
                    MPDRNN_PATHS.get_data_path("metrics_isolete"),
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_isolete"),
            },
            "ipmpdrnn": {
                "path_to_cm":
                    IPMPDRNN_PATHS.get_data_path("cm_isolete"),
                "path_to_metrics":
                    IPMPDRNN_PATHS.get_data_path("metrics_isolete"),
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_isolete"),
            }
        },
        "letter": {
            "mpdrnn": {
                "path_to_cm":
                    MPDRNN_PATHS.get_data_path("cm_letter"),
                "path_to_metrics":
                    MPDRNN_PATHS.get_data_path("metrics_letter"),
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_letter"),
            },
            "ipmpdrnn": {
                "path_to_cm":
                    IPMPDRNN_PATHS.get_data_path("cm_letter"),
                "path_to_metrics":
                    IPMPDRNN_PATHS.get_data_path("metrics_letter"),
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_letter"),
            }
        },
        "mnist": {
            "mpdrnn": {
                "path_to_cm":
                    MPDRNN_PATHS.get_data_path("cm_mnist"),
                "path_to_metrics":
                    MPDRNN_PATHS.get_data_path("metrics_mnist"),
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_mnist"),
            },
            "ipmpdrnn": {
                "path_to_cm":
                    IPMPDRNN_PATHS.get_data_path("cm_mnist"),
                "path_to_metrics":
                    IPMPDRNN_PATHS.get_data_path("metrics_mnist"),
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_mnist"),
            }
        },
        "mnist_fashion": {
            "mpdrnn": {
                "path_to_cm":
                    MPDRNN_PATHS.get_data_path("cm_mnist_fashion"),
                "path_to_metrics":
                    MPDRNN_PATHS.get_data_path("metrics_mnist_fashion"),
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_mnist_fashion")
            },
            "ipmpdrnn": {
                "path_to_cm":
                    IPMPDRNN_PATHS.get_data_path("cm_mnist_fashion"),
                "path_to_metrics":
                    IPMPDRNN_PATHS.get_data_path("metrics_mnist_fashion"),
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_mnist_fashion")
            }
        },
        "musk2": {
            "mpdrnn": {
                "path_to_cm":
                    MPDRNN_PATHS.get_data_path("cm_musk2"),
                "path_to_metrics":
                    MPDRNN_PATHS.get_data_path("metrics_musk2"),
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_musk2")
            },
            "ipmpdrnn": {
                "path_to_cm":
                    IPMPDRNN_PATHS.get_data_path("cm_musk2"),
                "path_to_metrics":
                    IPMPDRNN_PATHS.get_data_path("metrics_musk2"),
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_musk2")
            }
        },
        "optdigits": {
            "mpdrnn": {
                "path_to_cm":
                    MPDRNN_PATHS.get_data_path("cm_optdigits"),
                "path_to_metrics":
                    MPDRNN_PATHS.get_data_path("metrics_optdigits"),
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_optdigits"),
            },
            "ipmpdrnn": {
                "path_to_cm":
                    IPMPDRNN_PATHS.get_data_path("cm_optdigits"),
                "path_to_metrics":
                    IPMPDRNN_PATHS.get_data_path("metrics_optdigits"),
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_optdigits"),
            }
        },
        "page_blocks": {
            "mpdrnn": {
                "path_to_cm":
                    MPDRNN_PATHS.get_data_path("cm_page_blocks"),
                "path_to_metrics":
                    MPDRNN_PATHS.get_data_path("metrics_page_blocks"),
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_page_blocks")
            },
            "ipmpdrnn": {
                "path_to_cm":
                    IPMPDRNN_PATHS.get_data_path("cm_page_blocks"),
                "path_to_metrics":
                    IPMPDRNN_PATHS.get_data_path("metrics_page_blocks"),
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_page_blocks")
            }
        },
        "satimages": {
            "mpdrnn": {
                "path_to_cm":
                    MPDRNN_PATHS.get_data_path("cm_satimages"),
                "path_to_metrics":
                    MPDRNN_PATHS.get_data_path("metrics_satimages"),
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_satimages")
            },
            "ipmpdrnn": {
                "path_to_cm":
                    IPMPDRNN_PATHS.get_data_path("cm_satimages"),
                "path_to_metrics":
                    IPMPDRNN_PATHS.get_data_path("metrics_satimages"),
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_satimages")
            }
        },
        "segment": {
            "mpdrnn": {
                "path_to_cm":
                    MPDRNN_PATHS.get_data_path("cm_segment"),
                "path_to_metrics":
                    MPDRNN_PATHS.get_data_path("metrics_segment"),
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_segment")
            },
            "ipmpdrnn": {
                "path_to_cm":
                    IPMPDRNN_PATHS.get_data_path("cm_segment"),
                "path_to_metrics":
                    IPMPDRNN_PATHS.get_data_path("metrics_segment"),
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_segment")
            }
        },
        "shuttle": {
            "mpdrnn": {
                "path_to_cm":
                    MPDRNN_PATHS.get_data_path("cm_shuttle"),
                "path_to_metrics":
                    MPDRNN_PATHS.get_data_path("metrics_shuttle"),
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_shuttle"),
            },
            "ipmpdrnn": {
                "path_to_cm":
                    IPMPDRNN_PATHS.get_data_path("cm_shuttle"),
                "path_to_metrics":
                    IPMPDRNN_PATHS.get_data_path("metrics_shuttle"),
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_shuttle"),
            }
        },
        "spambase": {
            "mpdrnn": {
                "path_to_cm":
                    MPDRNN_PATHS.get_data_path("cm_spambase"),
                "path_to_metrics":
                    MPDRNN_PATHS.get_data_path("metrics_spambase"),
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_spambase")
            },
            "ipmpdrnn": {
                "path_to_cm":
                    IPMPDRNN_PATHS.get_data_path("cm_spambase"),
                "path_to_metrics":
                    IPMPDRNN_PATHS.get_data_path("metrics_spambase"),
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_spambase"),
            }
        },
        "usps": {
            "mpdrnn": {
                "path_to_cm":
                    MPDRNN_PATHS.get_data_path("cm_usps"),
                "path_to_metrics":
                    MPDRNN_PATHS.get_data_path("metrics_usps"),
                "path_to_results":
                    MPDRNN_PATHS.get_data_path("results_usps")
            },
            "ipmpdrnn": {
                "path_to_cm":
                    IPMPDRNN_PATHS.get_data_path("cm_usps"),
                "path_to_metrics":
                    IPMPDRNN_PATHS.get_data_path("metrics_usps"),
                "path_to_results":
                    IPMPDRNN_PATHS.get_data_path("results_usps")
            }
        }
    }

    if dataset_type not in dataset_config:
        raise ValueError(f'Invalid dataset name: {dataset_type}')

    return dataset_config[dataset_type]


def fcnn_paths_configs(dataset_type) -> Dict:
    dataset_config = {
        "connect4": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_connect4"),
            "logs":
                FCNN_PATHS.get_data_path("logs_connect4"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_connect4")
        },
        "isolete": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_isolete"),
            "logs":
                FCNN_PATHS.get_data_path("logs_isolete"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_isolete")
        },
        "letter": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_letter"),
            "logs":
                FCNN_PATHS.get_data_path("logs_letter"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_letter")
        },
        "mnist": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_mnist"),
            "logs":
                FCNN_PATHS.get_data_path("logs_mnist"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_mnist")
        },
        "mnist_fashion": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_mnist_fashion"),
            "logs":
                FCNN_PATHS.get_data_path("logs_mnist_fashion"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_mnist_fashion")
        },
        "musk2": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_musk2"),
            "logs":
                FCNN_PATHS.get_data_path("logs_musk2"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_musk2")
        },
        "optdigits": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_optdigits"),
            "logs":
                FCNN_PATHS.get_data_path("logs_optdigits"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_optdigits")
        },
        "page_blocks": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_page_blocks"),
            "logs":
                FCNN_PATHS.get_data_path("logs_page_blocks"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_page_blocks")
        },
        "satimages": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_satimages"),
            "logs":
                FCNN_PATHS.get_data_path("logs_satimages"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_satimages")
        },
        "segment": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_segment"),
            "logs":
                FCNN_PATHS.get_data_path("logs_segment"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_segment")
        },
        "shuttle": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_shuttle"),
            "logs":
                FCNN_PATHS.get_data_path("logs_shuttle"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_shuttle")
        },
        "spambase": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_spambase"),
            "logs":
                FCNN_PATHS.get_data_path("logs_spambase"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_spambase")
        },
        "usps": {
            "fcnn_saved_weights":
                FCNN_PATHS.get_data_path("sw_usps"),
            "logs":
                FCNN_PATHS.get_data_path("logs_usps"),
            "saved_results":
                FCNN_PATHS.get_data_path("results_usps")
        }
    }

    if dataset_type not in dataset_config:
        raise ValueError(f"Invalid dataset name: {dataset_type}")

    return dataset_config[dataset_type]
