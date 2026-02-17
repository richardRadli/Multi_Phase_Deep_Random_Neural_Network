import colorlog
import json
import jsonschema
import logging
import matplotlib.pyplot as plt
import openpyxl
import pandas as pd
import numpy as np
import time
import torch
import sys

from datetime import datetime
from functools import wraps
from jsonschema import validate
from nn.dataloaders.npz_dataloader import NpzDataset
from sklearn.decomposition import PCA
from openpyxl.styles import PatternFill
from torch.utils.data import DataLoader
from typing import Any, Callable, List, Tuple, Union


def average_columns_in_excel(filename: str) -> None:
    """
    Calculates the average of each numeric column for each sheet in an Excel file and appends these averages to the end
    of each sheet.

    Args:
        filename (str): The path to the Excel file.

    Returns:
        None: The function modifies the Excel file in-place.
    """

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


def create_timestamp() -> str:
    """
    Creates a timestamp in the format of '%Y-%m-%d_%H-%M-%S', representing the current date and time.

    Returns:
        Timestamp string in the format of '%Y-%m-%d_%H-%M-%S'.
    """

    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def create_train_valid_test_datasets(file_path, batch_size=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoader instances for training, validation, and testing datasets from a given file path.

    Args:
        file_path (str): The path to the file from which the datasets will be created.
        batch_size (int): The batch size to use for training. If None, the default whole dataset will be used.

    Returns:
        tuple: A tuple containing three DataLoader instances:
            - `train_loader`: DataLoader for the training dataset.
            - `valid_loader`: DataLoader for the validation dataset.
            - `test_loader`: DataLoader for the testing dataset.
    """

    train_dataset = NpzDataset(file_path, operation="train")
    valid_dataset = NpzDataset(file_path, operation="valid")
    test_dataset = NpzDataset(file_path, operation="test")

    train_loader = (
        DataLoader(
            dataset=train_dataset, batch_size=len(train_dataset) if batch_size is None else batch_size, shuffle=False
        )
    )
    valid_loader = (
        DataLoader(
            dataset=valid_dataset, batch_size=len(valid_dataset) if batch_size is None else batch_size, shuffle=False
        )
    )
    test_loader = (
        DataLoader(
            dataset=test_dataset, batch_size=len(test_dataset) if batch_size is None else batch_size, shuffle=False
        )
    )

    logging.info(f"Size of train dataset: {len(train_dataset)}, Size of test dataset: {len(test_dataset)}")

    return train_loader, valid_loader, test_loader


def insert_data_to_excel(filename: str, dataset_name: str, row: int, data: list) -> None:
    """
    Inserts a row of data into a specific sheet of an Excel workbook, creating the sheet if it doesn't exist.

    Args:
        filename (str): The path to the Excel file where data will be inserted.
        dataset_name (str): The name of the sheet in which to insert the data.
        row (int): The row number in the sheet where the data should be inserted.
        data (List[List[Union[str, int, float]]]): A list containing rows of data to be inserted into the specified row.
            Each row is a list of values, which can be strings, integers, or floats.

    Returns:
        None: The function modifies the Excel file in-place.
    """

    try:
        workbook = openpyxl.load_workbook(filename)
    except FileNotFoundError:
        workbook = openpyxl.Workbook()

    if dataset_name not in workbook.sheetnames:
        workbook.create_sheet(dataset_name)

    sheet = workbook[dataset_name]

    values = ["train acc",
              "test acc",
              "train precision",
              "test precision",
              "train recall",
              "test recall",
              "train f1",
              "test f1",
              "training time"]

    for col, value in enumerate(values, start=1):
        sheet.cell(row=1, column=col, value=value)

    for col, value in enumerate(data[0], start=1):
        sheet.cell(row=row, column=col, value=str(value))

    if "Sheet" in workbook.sheetnames:
        del workbook['Sheet']

    workbook.save(filename)


def load_config_json(json_schema_filename: str, json_filename: str):
    """
    Loads and validates a JSON configuration file against a JSON schema, flattens and processes the configuration,
    and returns the processed configuration as a dictionary.

    Args:
        json_schema_filename (str): The path to the JSON schema file used for validation.
        json_filename (str): The path to the JSON configuration file to be validated and processed.

    Returns:
        Dict[str, Any]: A dictionary containing the processed configuration. Keys are configuration parameters,
                        and values are their corresponding values.

    Raises:
        jsonschema.exceptions.ValidationError: If the JSON data does not conform to the schema.
    """

    with open(json_schema_filename, "r") as schema_file:
        schema = json.load(schema_file)

    with open(json_filename, "r") as config_file:
        config = json.load(config_file)

    try:
        validate(config, schema)
        logging.info("JSON data is valid.")

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', None)

        flattened_config = {}

        simple_config = {k: v for k, v in config.items() if not isinstance(v, dict)}
        nested_config = {k: v for k, v in config.items() if isinstance(v, dict)}

        dataset_name = simple_config['dataset_name']

        if 'hyperparamtuning' in nested_config:
            flattened_config['hyperparamtuning'] = nested_config['hyperparamtuning']

        if 'optimization' in nested_config:
            flattened_config['optimization'] = nested_config['optimization']

        for key, value in nested_config.items():
            if key != 'hyperparamtuning' and dataset_name in value:
                flattened_config[key] = value[dataset_name]

        full_config = {**simple_config, **flattened_config}
        df = pd.DataFrame.from_dict(full_config, orient='index', columns=['Value'])

        logging.info("Config DataFrame:\n" + df.to_string())

        return full_config
    except jsonschema.exceptions.ValidationError as err:
        logging.error(f"JSON data is invalid: {err}")


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


def plot_condition_number(cond_list, save_path):
    layers = ('beta', 'gamma', 'delta')
    cond_number_counts = {
        'cond_number': np.array([cond_list[0+1], cond_list[2+1], cond_list[4+1]]),
    }
    width = 0.6

    fig, ax = plt.subplots()
    bottom = np.zeros(3)

    for cn, condition_num_count in cond_number_counts.items():
        p = ax.bar(layers, condition_num_count, width, label=cn, bottom=bottom)
        bottom += condition_num_count

        ax.bar_label(p, label_type='center')

    ax.set_title('Condition number of each output layer')
    ax.legend()

    plt.savefig(save_path, dpi=200)

def plot_weights_histogram(hidden_layers, save_path):
    values = []
    for hidden_layer in hidden_layers:
        value = hidden_layer.numpy().flatten()
        values.append([value])

    plt.figure(figsize=(16, 6))
    for idx, value in enumerate(values, start=1):
        plt.subplot(1, 3, idx)
        plt.hist(value, bins=50)
        plt.xlabel("Activation value")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of hidden layer {idx}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)


def plot_vector_diversity(weights_list, save_path):
    plt.figure(figsize=(16, 6))

    for idx, weight in enumerate(weights_list):
        if isinstance(weight, (list, tuple)):
            weight = torch.cat(weight, dim=1)

        normed_weights = weight / (torch.linalg.norm(weight, dim=0, keepdim=True) + 1e-8)
        sim = (normed_weights.T @ normed_weights).detach().cpu().numpy()
        plt.subplot(1, 3, idx+1)
        plt.imshow(sim, cmap="viridis", vmin=-1, vmax=1)
        plt.colorbar()
        plt.title(f"Weights in layer {idx+1}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def plot_neuron_vectors_3d(vectors_list, save_path=None):
    fig = plt.figure(figsize=(16, 6))

    axes = [
        fig.add_subplot(1, 3, i + 1, projection='3d')
        for i in range(3)
    ]

    for idx, vector in enumerate(vectors_list):
        if isinstance(vector, (list, tuple)):
            vector = torch.cat(vector, dim=1)

        W = vector.T.detach().cpu().numpy()

        if W.shape[1] > 3:
            W_3d = PCA(n_components=3).fit_transform(W)
        else:
            W_3d = W

        ax = axes[idx]
        ax.scatter(W_3d[:, 0], W_3d[:, 1], W_3d[:, 2], c="blue", s=50)
        ax.set_title(f"Neuron directions {idx + 1}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


def reorder_metrics_lists(train_metrics, test_metrics, training_time_list = None) -> List:
    """
    Reorders and combines training and testing metrics into a single list of metrics.

    Args:
        train_metrics:
            A list of training metrics.
            Expected order: [train_acc, train_precision, train_recall, train_f1] or
            [train_acc, train_precision, train_recall, train_f1, training_time] if training_time_list is not provided.
        test_metrics:
            A list of testing metrics.
            Expected order: [test_acc, test_precision, test_recall, test_f1].
        training_time_list (Optional[List[Union[int, float]]]):
            An optional list of training times. If provided, its sum is used as the total training time.

    Returns:
        List: A list containing a single tuple with the reordered and combined metrics.

    Notes:
        - If `training_time_list` is provided, the function sums it and appends it to the combined metrics.
        - If `training_time_list` is not provided, the training time is taken from the `train_metrics` list.
    """

    if training_time_list is not None:
        training_time = round(sum(training_time_list), 3)

        train_acc, train_precision, train_recall, train_f1 = (
            round(train_metrics[0], 3),
            round(train_metrics[1], 3),
            round(train_metrics[2], 3),
            round(train_metrics[3], 3),
        )
    else:
        train_acc, train_precision, train_recall, train_f1, training_time = (
            round(train_metrics[0], 3),
            round(train_metrics[1], 3),
            round(train_metrics[2], 3),
            round(train_metrics[3], 3),
            round(train_metrics[4], 3),
        )

    test_acc, test_precision, test_recall, test_f1 = (
        round(test_metrics[0], 3),
        round(test_metrics[1], 3),
        round(test_metrics[2], 3),
        round(test_metrics[3], 3),
    )

    combined_metrics = [
        train_acc, test_acc,
        train_precision, test_precision,
        train_recall, test_recall,
        train_f1, test_f1,
        training_time
    ]

    return [tuple(combined_metrics)]


def save_log_to_txt(output_file: str, result: Any, operation: str) -> None:
    """
    Saves the best trial configuration and results to a text file based on the specified operation.

    Args:
        output_file (str): The path to the output text file where the log will be saved.
        result (Any): An object containing the results of the trials. This should have a `get_best_trial` method to
            retrieve the best trial based on the given criteria.
        operation (str): A string specifying the type of operation to log. Should be either "loss" or "accuracy".

    Returns:
        None: The function writes the log to the specified file and does not return any value.

    Raises:
        ValueError: If the provided operation is not "loss" or "accuracy".
    """

    original_stdout = sys.stdout

    with open(output_file, "w") as log_file:
        sys.stdout = log_file

        if operation == "loss":
            best_trial = result.get_best_trial("loss", "min", "last")
            print("Best trial config: {}".format(best_trial.config))
            print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
            print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))
        elif operation == "accuracy":
            best_trial = result.get_best_trial("accuracy", "max", "last")
            print("Best trial config: {}".format(best_trial.config))
            print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))
        else:
            raise ValueError(f"Invalid operation: {operation}")

    sys.stdout = original_stdout

    logging.info(f"Saving log to {output_file}")


def setup_logger() -> logging.Logger:
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

    logger = logging.getLogger()
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        })

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger