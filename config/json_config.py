from config.data_paths import JSON_FILES_PATHS


def json_config_selector(network):
    json_cfg = {
        "fcnn": {
            "config": JSON_FILES_PATHS.get_data_path("config_fcnn"),
            "schema": JSON_FILES_PATHS.get_data_path("config_schema_fcnn")
        },
        "helm": {
            "config": JSON_FILES_PATHS.get_data_path("config_helm"),
            "schema": JSON_FILES_PATHS.get_data_path("config_schema_helm")
        },
        "mpdrnn": {
            "config": JSON_FILES_PATHS.get_data_path("config_mpdrnn"),
            "schema": JSON_FILES_PATHS.get_data_path("config_schema_mpdrnn")
        },
        "ipmpdrnn": {
            "config": JSON_FILES_PATHS.get_data_path("config_ipmpdrnn"),
            "schema": JSON_FILES_PATHS.get_data_path("config_schema_ipmpdrnn")
        },
        "cipmpdrnn": {
            "config": JSON_FILES_PATHS.get_data_path("config_cipmpdrnn"),
            "schema": JSON_FILES_PATHS.get_data_path("config_schema_cipmpdrnn")
        },
    }

    return json_cfg[network]