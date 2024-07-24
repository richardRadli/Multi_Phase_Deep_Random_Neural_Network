import colorlog
import gc
import json
import jsonschema
import logging
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import os
import pandas as pd
import seaborn as sns
import time
import torch
import sys

from datetime import datetime
from functools import wraps
from jsonschema import validate
from openpyxl.styles import PatternFill
from pprint import pformat
from typing import Any, Callable


def setup_logger():
    """
    Set up a colorized logger with the following log levels and colors:

    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Red on a white background

    Returns:
        The configured logger instance.
    """

    # Check if logger has already been set up
    logger = logging.getLogger()
    if logger.hasHandlers():
        return logger

    # Set up logging
    logger.setLevel(logging.INFO)

    # Create a colorized formatter
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        })

    # Create a console handler and add the formatter to it
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- C R E A T E   T I M E S T A M P ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def create_timestamp() -> str:
    """
    Creates a timestamp in the format of '%Y-%m-%d_%H-%M-%S', representing the current date and time.

    :return: The timestamp string.
    """

    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def create_dir(root_dir: str, method: str):
    timestamp = create_timestamp()
    output_dir = os.path.join(root_dir, f"{timestamp}_{method}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------- P R E T T Y   P R I N T   R E S U L T S --------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def pretty_print_results(acc, precision, recall, fscore, loss, root_dir: str, operation: str, name: str,
                         exe_time=None) -> None:
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    columns = [
        f"{operation} loss",
        f"{operation} acc",
        f"{operation} precision",
        f"{operation} recall",
        f"{operation} fscore",
    ]

    data = [
        np.round(loss, 4),
        np.round(acc, 4),
        np.round(precision, 4),
        np.round(recall, 4),
        np.round(fscore, 4)
    ]

    if exe_time is not None:
        columns.append(f"{operation} exe time")
        data.append(np.round(exe_time, 4))

    df = pd.DataFrame([data],
                      index=pd.Index(['Score']),
                      columns=pd.MultiIndex.from_product([columns]))

    path_to_save = os.path.join(root_dir, f"{operation}.txt")
    try:
        with open(path_to_save, 'a'):
            pass
    except FileNotFoundError:
        df.to_csv(path_to_save, sep='\t', index=True)
    else:
        df.to_csv(path_to_save, sep='\t', index=True, mode='a', header=True)

    upper_separator = "-" * 34  # Customize the separator as needed
    lower_separator = "-" * 83

    logging.info("\n%s %s %s %s \n%s\n%s", upper_separator, name, operation, upper_separator, df, lower_separator)


def measure_execution_time(func: Callable) -> Callable:
    """
    Decorator to measure the execution time of a function.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        """
        Wrapper function to measure execution time.

        Args:
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            Any: The result of the function.
        """

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        wrapper.execution_time = end_time - start_time
        logging.info(f"Execution time of {func.__name__}: {wrapper.execution_time:.4f} seconds")
        return result

    wrapper.execution_time = None
    return wrapper


def calc_exp_neurons(total_neurons: int, n_layers: int):
    # Calculate the geometric mean of the number of neurons based on the number of layers
    geom_mean = total_neurons ** (1 / n_layers)

    # Calculate the number of neurons for each layer using the geometric mean
    neurons_per_layer = [int(geom_mean ** i) for i in range(n_layers)]

    # Adjust the first layer to match the total number of neurons
    neurons_per_layer[0] += total_neurons - sum(neurons_per_layer)

    # Make sure the last layer has more than 1 neuron
    if neurons_per_layer[-1] <= 1:
        neurons_per_layer[-2] += neurons_per_layer[-1] - 1
        neurons_per_layer.pop()

    return sorted(neurons_per_layer, reverse=True)


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- P L O T   C O N F   M T X ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def plot_confusion_matrix_mpdrnn(cm, path_to_plot, name_of_dataset: str, operation: str, method: str, labels=None) -> None:
    fig, axis = plt.subplots(1, 3, figsize=(15, 5))

    for i, cm in enumerate(cm):
        ax = axis[i]
        sns.heatmap(cm, annot=True, fmt='.0f', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title('%s, %s, %s, %s' % (f"Phase {i + 1}", name_of_dataset, method, operation))
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    filename = os.path.join(path_to_plot, f"{name_of_dataset}_{method}_{operation}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    gc.collect()


def plot_confusion_matrix_fcnn(cm, operation, class_labels, dataset_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted labels")
    plt.ylabel("Actual labels")
    plt.title(f"Confusion matrix of {dataset_name} on the {operation} set.")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------- P L O T   T R A I N   A N D   V A L I D   D A T A ----------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def plot_metrics(train, test, metric_type: str, path_to_plot: str, name_of_dataset: str, method: str) \
        -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    x_values = np.arange(1, len(train) + 1)

    filename = os.path.join(path_to_plot, f"{name_of_dataset}_{method}_{metric_type}.png")

    plt.plot(x_values, train, marker='o', label='Train')
    plt.plot(x_values, test, marker='o', label='Test')
    plt.xlabel('Phases')
    plt.ylabel(metric_type)
    plt.title('%s, %s, %s' % (name_of_dataset, method, metric_type))
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    gc.collect()


def device_selector(preferred_device) -> torch.device:
    """
    Provides information about the currently available GPUs and returns a torch device for training and inference.

    :return: A torch device for either "cuda" or "cpu".
    """
    if preferred_device not in ["cuda", "cpu"]:
        logging.warning("Preferred device is not valid. Using CPU instead.")
        return torch.device("cpu")

    if preferred_device == "cuda" and torch.cuda.is_available():
        cuda_info = {
            'CUDA Available': [torch.cuda.is_available()],
            'CUDA Device Count': [torch.cuda.device_count()],
            'Current CUDA Device': [torch.cuda.current_device()],
            'CUDA Device Name': [torch.cuda.get_device_name(0)]
        }

        df = pd.DataFrame(cuda_info)
        logging.info(df)
        return torch.device("cuda")

    if preferred_device in ["cuda"] and not torch.cuda.is_available():
        logging.info("Only CPU is available!")
        return torch.device("cpu")

    if preferred_device == "cpu":
        logging.info("Selected CPU device")
        return torch.device("cpu")


def insert_data_to_excel(filename, dataset_name, row, data):
    try:
        workbook = openpyxl.load_workbook(filename)
    except FileNotFoundError:
        workbook = openpyxl.Workbook()

    if dataset_name not in workbook.sheetnames:
        workbook.create_sheet(dataset_name)

    sheet = workbook[dataset_name]

    values = ["train acc", "test acc", "train precision", "test precision", "train recall", "test recall", "train f1",
              "test f1", "training time"]

    for col, value in enumerate(values, start=1):
        sheet.cell(row=1, column=col, value=value)

    for col, value in enumerate(data[0], start=1):
        sheet.cell(row=row, column=col, value=str(value))

    if "Sheet" in workbook.sheetnames:
        del workbook['Sheet']

    workbook.save(filename)


def average_columns_in_excel(filename: str):
    excel_file = pd.ExcelFile(filename)

    results = {}

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(filename, sheet_name=sheet_name)
        column_averages = df.mean(numeric_only=True)
        results[sheet_name] = column_averages

    workbook = openpyxl.load_workbook(filename)

    for sheet_name, avg in results.items():
        sheet = workbook[sheet_name]
        first_empty_row = sheet.max_row + 1

        for col_num, (col_name, value) in enumerate(avg.items(), start=1):
            sheet.cell(row=first_empty_row, column=col_num, value=value)

        fill = PatternFill(start_color="00CCFF", end_color="00CCFF", fill_type="solid")
        for cell in sheet[first_empty_row]:
            cell.fill = fill

    workbook.save(filename)


def load_config_json(json_schema_filename: str, json_filename: str):
    with open(json_schema_filename, "r") as schema_file:
        schema = json.load(schema_file)

    with open(json_filename, "r") as config_file:
        config = json.load(config_file)

    try:
        validate(config, schema)
        logging.info("JSON data is valid.")

        # Adjust Pandas display settings
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', None)

        # Split config into simple and nested dictionaries
        simple_config = {k: v for k, v in config.items() if not isinstance(v, dict)}
        nested_config = {k: v for k, v in config.items() if isinstance(v, dict)}

        # Create DataFrame for simple config
        df = pd.DataFrame.from_dict(simple_config, orient='index')

        # Pretty print the DataFrame
        logging.info("Simple Config DataFrame:\n" + df.to_string())

        # Pretty print nested structures separately
        for key, value in nested_config.items():
            logging.info(f"{key}:\n{pformat(value)}")

        # Merge simple and nested configs back for returning
        full_config = {**simple_config, **nested_config}

        return full_config
    except jsonschema.exceptions.ValidationError as err:
        logging.error(f"JSON data is invalid: {err}")


def find_latest_file_in_latest_directory(path: str) -> str:
    dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    if not dirs:
        raise ValueError(f"No directories found in {path}")

    dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_dir = dirs[0]
    files = [os.path.join(latest_dir, f) for f in os.listdir(latest_dir) if
             os.path.isfile(os.path.join(latest_dir, f))]

    if not files:
        raise ValueError(f"No files found in {latest_dir}")

    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_file = files[0]
    logging.info(f"Latest file found: {latest_file}")

    return latest_file


def save_log_to_txt(output_file, result):
    original_stdout = sys.stdout

    with open(output_file, "w") as log_file:
        sys.stdout = log_file

        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

    sys.stdout = original_stdout

    logging.info(f"Saving log to {output_file}")


def reorder_metrics_lists(train_metrics, test_metrics, training_time_list=None):
    if training_time_list is not None:
        training_time = sum(training_time_list)

        train_acc, train_precision, train_recall, train_f1, = (
            train_metrics[0], train_metrics[1], train_metrics[2], train_metrics[3]
        )
    else:
        train_acc, train_precision, train_recall, train_f1, training_time = (
            train_metrics[0], train_metrics[1], train_metrics[2], train_metrics[3], train_metrics[4]
        )

    test_acc, test_precision, test_recall, test_f1 = (
        test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3]
    )

    combined_metrics = [
        train_acc, test_acc,
        train_precision, test_precision,
        train_recall, test_recall,
        train_f1, test_f1,
        training_time
    ]

    return [tuple(combined_metrics)]


def find_best_testing_accuracy(accuracies):
    best_metrics = max(accuracies, key=lambda x: x[1][0])
    return best_metrics
