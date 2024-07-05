import os

from typing import Dict

from config.const import (ELM_DATASET_FILES_PATHS, MPDRNN_PATHS)


def elm_general_dataset_configs(cfg) -> Dict:
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
            "dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_connect4"), "data.txt"),
            "cached_dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_connect4"), "connect4.npz"),
            "path_to_cm":
                MPDRNN_PATHS.get_data_path("cm_connect4"),
            "path_to_metrics":
                MPDRNN_PATHS.get_data_path("metrics_connect4"),
            "path_to_results":
                MPDRNN_PATHS.get_data_path("results_connect4"),
        },
        "forest": {
            "dataset_name":
                "forest",
            "num_train_data":
                15120,
            "num_test_data":
                565892,
            "num_features":
                54,
            "num_classes":
                7,
            "eq_neurons":
                [1000, 1000, 1000],
            "exp_neurons":
                [100, 40, 10],
            "helm_neurons":
                [50, 50, 51],
            "class_labels":
                ["Spruce", "Lodgepole-Pine", "Ponderosa-Pine", "Cottonwood", "Aspen", "Douglas-fir", "Krummholz"],
            "dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_forest"), "data.txt"),
            "cached_dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_forest"), "forest.npz"),
            "path_to_cm":
                MPDRNN_PATHS.get_data_path("cm_forest"),
            "path_to_metrics":
                MPDRNN_PATHS.get_data_path("metrics_forest"),
            "path_to_results":
                MPDRNN_PATHS.get_data_path("results_forest"),
        },
        "iris": {
            "dataset_name":
                "iris",
            "num_train_data":
                105,
            "num_test_data":
                45,
            "num_features":
                4,
            "num_classes":
                3,
            "eq_neurons":
                [100, 100, 100],
            "exp_neurons":
                [100, 40, 10],
            "helm_neurons":
                [50, 50, 51],
            "class_labels":
                ['iris-setosa', 'iris-versicolor', 'iris-virginica'],
            "dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_iris"), "data.txt"),
            "cached_dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_iris"), "iris.npz"),
            "path_to_cm":
                MPDRNN_PATHS.get_data_path("cm_iris"),
            "path_to_metrics":
                MPDRNN_PATHS.get_data_path("metrics_iris"),
            "path_to_results":
                MPDRNN_PATHS.get_data_path("results_iris"),
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
            "helm_neurons":
                [800, 400, 3000],
            "class_labels":
                [],
            "dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_isolete"), "data.txt"),
            "cached_dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_isolete"), "isolete.npz"),
            "path_to_cm":
                MPDRNN_PATHS.get_data_path("cm_isolete"),
            "path_to_metrics":
                MPDRNN_PATHS.get_data_path("metrics_isolete"),
            "path_to_results":
                MPDRNN_PATHS.get_data_path("results_isolete"),
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
            "dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_letter"), "data.txt"),
            "cached_dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_letter"), "letter.npz"),
            "path_to_cm":
                MPDRNN_PATHS.get_data_path("cm_letter"),
            "path_to_metrics":
                MPDRNN_PATHS.get_data_path("metrics_letter"),
            "path_to_results":
                MPDRNN_PATHS.get_data_path("results_letter"),
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
            "helm_neurons":
                [1000, 5000, 10000],
            "class_labels":
                ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_mnist"), "data.txt"),
            "cached_dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_mnist"), "mnist.npz"),
            "path_to_cm":
                MPDRNN_PATHS.get_data_path("cm_mnist"),
            "path_to_metrics":
                MPDRNN_PATHS.get_data_path("metrics_mnist"),
            "path_to_results":
                MPDRNN_PATHS.get_data_path("results_mnist"),
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
            "dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_mnist_fashion"), "data.txt"),
            "cached_dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_mnist_fashion"), "mnist_fashion.npz"),
            "path_to_cm":
                MPDRNN_PATHS.get_data_path("cm_mnist_fashion"),
            "path_to_metrics":
                MPDRNN_PATHS.get_data_path("metrics_mnist_fashion"),
            "path_to_results":
                MPDRNN_PATHS.get_data_path("results_mnist_fashion"),
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
            "dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_musk2"), "data.txt"),
            "cached_dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_musk2"), "musk2.npz"),
            "path_to_cm":
                MPDRNN_PATHS.get_data_path("cm_musk2"),
            "path_to_metrics":
                MPDRNN_PATHS.get_data_path("metrics_musk2"),
            "path_to_results":
                MPDRNN_PATHS.get_data_path("results_musk2"),
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
            "helm_neurons":
                [100, 50, 500],
            "class_labels":
                ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_optdigits"), "data.txt"),
            "cached_dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_optdigits"), "optdigits.npz"),
            "path_to_cm":
                MPDRNN_PATHS.get_data_path("cm_optdigits"),
            "path_to_metrics":
                MPDRNN_PATHS.get_data_path("metrics_optdigits"),
            "path_to_results":
                MPDRNN_PATHS.get_data_path("results_optdigits"),
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
            "helm_neurons":
                [100, 50, 500],
            "class_labels":
                ["text", "horiz. line", "graphic", "vert. line ", "picture"],
            "dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_page_blocks"), "data.txt"),
            "cached_dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_page_blocks"), "page_blocks.npz"),
            "path_to_cm":
                MPDRNN_PATHS.get_data_path("cm_page_blocks"),
            "path_to_metrics":
                MPDRNN_PATHS.get_data_path("metrics_page_blocks"),
            "path_to_results":
                MPDRNN_PATHS.get_data_path("results_page_blocks"),
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
            "helm_neurons":
                [50, 50, 51],
            "class_labels":
                [],
            "dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_satimages"), "data.txt"),
            "cached_dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_satimages"), "satimages.npz"),
            "path_to_cm":
                MPDRNN_PATHS.get_data_path("cm_satimages"),
            "path_to_metrics":
                MPDRNN_PATHS.get_data_path("metrics_satimages"),
            "path_to_results":
                MPDRNN_PATHS.get_data_path("results_satimages")
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
            "dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_segment"), "data.txt"),
            "cached_dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_segment"), "segment.npz"),
            "path_to_metrics":
                MPDRNN_PATHS.get_data_path("metrics_segment"),
            "path_to_results":
                MPDRNN_PATHS.get_data_path("results_segment"),
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
            "helm_neurons":
                [100, 50, 500],
            "class_labels":
                ["Rad Flow", "Fpv Close", "Fpv Open", "High", "Bypass", "Bpv Close", "Bpv Open"],
            "dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_shuttle"), "data.txt"),
            "cached_dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_shuttle"), "shuttle.npz"),
            "path_to_cm":
                MPDRNN_PATHS.get_data_path("cm_shuttle"),
            "path_to_metrics":
                MPDRNN_PATHS.get_data_path("metrics_shuttle"),
            "path_to_results":
                MPDRNN_PATHS.get_data_path("results_shuttle"),
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
            "dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_spambase"), "data.txt"),
            "cached_dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_spambase"), "spambase.npz"),
            "path_to_cm":
                MPDRNN_PATHS.get_data_path("cm_spambase"),
            "path_to_metrics":
                MPDRNN_PATHS.get_data_path("metrics_spambase"),
            "path_to_results":
                MPDRNN_PATHS.get_data_path("results_spambase"),
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
            "dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_usps"), "data.txt"),
            "cached_dataset_file":
                os.path.join(ELM_DATASET_FILES_PATHS.get_data_path("dataset_path_usps"), "usps.npz"),
            "path_to_cm":
                MPDRNN_PATHS.get_data_path("cm_usps"),
            "path_to_metrics":
                MPDRNN_PATHS.get_data_path("metrics_usps"),
            "path_to_results":
                MPDRNN_PATHS.get_data_path("results_usps")
        }
    }

    if dataset_type not in dataset_config:
        raise ValueError(f'Invalid dataset name: {dataset_type}')

    return dataset_config[dataset_type]
