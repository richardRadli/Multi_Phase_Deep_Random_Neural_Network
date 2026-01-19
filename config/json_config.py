from config.data_paths import JSON_FILES_PATHS


def json_config_selector(network):
    json_cfg = {
        "mpdrnn": {
            "config": JSON_FILES_PATHS.get_data_path("config_mpdrnn"),
            "schema": JSON_FILES_PATHS.get_data_path("config_schema_mpdrnn")
        },
        "dev_drnn": {
            "config": JSON_FILES_PATHS.get_data_path("config_dev_drnn"),
            "schema": JSON_FILES_PATHS.get_data_path("config_schema_dev_drnn")
        }
    }

    return json_cfg[network]
