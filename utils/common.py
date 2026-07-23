import colorlog
import json
import logging

import pandas as pd

from jsonschema import validate
import jsonschema


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


def load_config_json(json_schema_filename: str, json_filename: str):
    """
    Loads and validates a JSON configuration file against a JSON schema, flattens and processes the configuration,
    and returns the processed configuration as a dictionary.
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