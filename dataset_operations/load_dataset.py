import numpy as np

from scipy import stats
from sklearn import preprocessing


def load_data(gen_ds_cfg, cfg_data_preprocessing):
    data_file = gen_ds_cfg.get("cached_dataset_file")
    data = np.load(data_file, allow_pickle=True)
    
    train_data = data[0]
    train_labels = data[2]
    test_data = data[1]
    test_labels = data[3]
    
    if cfg_data_preprocessing.normalize:
        if cfg_data_preprocessing.type_of_normalization == "zscore":
            train_data = stats.zscore(train_data)
            test_data = stats.zscore(test_data)
        elif cfg_data_preprocessing.type_of_normalization == "minmax":
            scaler = preprocessing.MinMaxScaler()
            train_data = scaler.fit_transform(train_data)
            test_data = scaler.fit_transform(test_data)
        else:
            raise ValueError("Wrong type of normalization!")

    return train_data, train_labels, test_data, test_labels
