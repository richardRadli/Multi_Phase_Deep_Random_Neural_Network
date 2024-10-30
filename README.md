# Multi Phase Deep Random Neural Networks
 
![Python](https://img.shields.io/badge/python-v3.11-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.2.1-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit-v1.4.0--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-v1.26.4-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-v2.1.0-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-v1.12.0-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![Ray Badge](https://img.shields.io/badge/Ray-v2.23.0-028CF0?logo=ray&logoColor=fff&style=for-the-badge)


# Overview

## ✨ Features
- Multiple network types (FCNN, HELM, MPDRNN)
- Hyperparameter tuning for optimal performance
- Dataset conversion and integration
- Detailed configuration options via JSON files

## 🔧 Prerequisites
- Python 3.10 or higher
- Optional: CUDA-enabled GPU (for PyTorch with CUDA support)

## 📚 Datasets
Datasets that are employed in our articles can be found and downloaded on the following websites:

1. UCI Machine Learning Repository: <a href="https://www.example.com/my great page">UCI Repository</a>
2. Google Drive: <a href="https://drive.google.com/file/d/1Fe3DjPOGgzNmlnJj0yn0WTUazh3ojL8k/view?usp=drive_link">Download here</a>

In case of downloading from the UCI Machine Learning Repository, one must convert the datasets to an appropriate format. 
For that use the provided Python file _convert_datasets.py_. Datasets on the Google Drive link are already converted.

## 📝 Requirements
Make sure you have the following packages installed:

```
colorama~=0.4.6
colorlog~=6.8.2
jsonschema~=4.23.0
matplotlib~=3.8.1
numpy~=1.26.4
openpyxl~=3.1.2
pandas~=2.1.0
ray~=2.32.0
seaborn~=0.13.0
scipy~=1.12.0
sklearn~=1.4.0
torch~=2.2.1+cu121
torchvision~=0.17.1+cu121
tqdm~=4.66.2
torchinfo~=1.8.0
```

You can install the listed packages with the following command:
```bash
pip install -r requirements.txt
```

## 🚀 Installation

### 1. Clone or download the repository
Begin by cloning or downloading this repository to your local machine.

### 2. Update configuration
Open the _data_paths.py_ file. You will find the following dictionary:

```python
root_mapping = {
    'ricsi': {
        "STORAGE_ROOT":
            "D:/storage/Journal2",
        "DATASET_ROOT":
            "D:/storage/Journal2/datasets",
        "PROJECT_ROOT":
            "C:/Users/ricsi/Documents/research/Multi_Phase_Deep_Random_Neural_Network",
    }
}
```

You have to replace the username (in this case 'ricsi') with your own. Your username can be acquired by running the `whoami` command in your terminal to retrieve it.

#### STORAGE_ROOT: 
- Adjust this path to the location where you want to save project outputs and other data generated during the execution of the Python files.

#### DATASET_ROOT: 
- Modify this path to point to the directory where your datasets are stored. This folder should contain all datasets necessary for the project. It should look like this:

* D:\storage\Journal2\datasets
  * connect4
    * connect4.npz
    * data.txt

#### PROJECT_ROOT
- Update this path to the directory where the Python and JSON files of the project are located.

### 3. Create necessary folders
Run the __data_paths.py__ script. This will create all the required folders based on the paths specified in the configuration.

### 4. Download and place datasets
Obtain the necessary datasets and place them into the DATASET_ROOT directory as specified in your updated configuration.

## 💻 Usage
### Setting Up Configuration Files
Before running the Python scripts, you need to configure your settings by preparing the following JSON and Python files:
- Configuration for the FCNN (FCNN_config.json)
- Configuration for the HELM (HELM_config.json)
- Configuration for the MPDRNN (MPDRNN_config.json)


Once your configuration files are set up, run the Python scripts to train and test.

### Workflow
- Optional: It is advisable to run hyperparameter tuning, although config files contain the best settings.
  - There is a separate file for hyperparameter tuning for all available networks. 
- After tuning, you may execute the training and evaluation for the desired network.
  - To train and evaluate the FCNN network, run __execute_test.py__ in the fcnn folder.
    - FCNN can be individually trained (__train_fcnn.py__) and evaluated (__eval_fcnn.py__) 
  - To train and evaluate the HELM network, run __helm.py__ in the helm folder.
  - To train and evaluate the MPDRNN network, run __mpdrnn.py__ in the mpdrnn folder.


## 📰 Link to paper

For detailed insights, check out our [research paper](https://link.springer.com/chapter/10.1007/978-3-031-44505-7_9).