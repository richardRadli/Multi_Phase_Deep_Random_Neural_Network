import colorama
import os

from tqdm import tqdm

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import fcnn_paths_configs
from eval_fcnn import EvalFCNN
from train_fcnn import TrainFCNN
from utils.utils import create_timestamp, insert_data_to_excel, load_config_json, average_columns_in_excel


def main():
    timestamp = create_timestamp()
    colorama.init()

    cfg = (
        load_config_json(json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_fcnn"),
                         json_filename=JSON_FILES_PATHS.get_data_path("config_fcnn"))
    )
    dataset_name = cfg.get("dataset_name")
    fcnn_config = fcnn_paths_configs(dataset_name)

    filename = (
        os.path.join(
            fcnn_config.get("saved_results"),
            f"{timestamp}_bs_{cfg.get('batch_size').get(dataset_name)}_hn_{cfg.get('hidden_neurons').get(dataset_name)}"
            f"_lr_{cfg.get('learning_rate').get(dataset_name)}_device_{cfg.get('device')}.xlsx")
    )

    collected_data = []

    for i in tqdm(range(cfg.get("num_tests")), desc=f"{colorama.Fore.LIGHTBLUE_EX} Testing cycle"):
        train_fcnn = TrainFCNN()
        train_fcnn.fit()
        training_time = train_fcnn.fit.execution_time

        eval_fcnn = EvalFCNN()
        eval_fcnn.main()
        collected_data.append((eval_fcnn.train_accuracy, eval_fcnn.test_accuracy,
                               eval_fcnn.train_precision, eval_fcnn.test_precision,
                               eval_fcnn.train_recall, eval_fcnn.test_recall,
                               eval_fcnn.train_f1sore, eval_fcnn.test_f1sore,
                               training_time))

        insert_data_to_excel(filename=filename,
                             dataset_name=dataset_name,
                             row=i + 2,
                             data=collected_data,
                             network="fcnn")

        collected_data.clear()

    average_columns_in_excel(filename)


if __name__ == '__main__':
    main()
