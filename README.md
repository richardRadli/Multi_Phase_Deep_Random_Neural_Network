![Python](https://img.shields.io/badge/python-v3.11-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.2.1-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit-v1.4.0--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-v1.26-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-v2.1.0-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-v1.12.0-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![Ray Badge](https://img.shields.io/badge/Ray-v2.23.0-028CF0?logo=ray&logoColor=fff&style=for-the-badge)


# Iteratively increasing randomized networks

## Datasets
Datasets that are employed in our article can be found on the website of 
<a href="https://www.example.com/my great page">UCI Machine Learning Repository</a>.

To convert them to appropriate format, use the python file called _convert_datasets.py_.

## Requirement
Make sure you have the following dependencies installed:

```bash
colorlog~=6.8.2
matplotlib~=3.8.1
numpy~=1.26.4
pandas~=2.1.0
sklearn==1.4.0
torch~=2.2.1+cu121
torchvision~=0.17.1+cu121
tqdm~=4.66.2
scikit-learn~=1.4.0
colorama~=0.4.6
jsonschema~=4.23.0
torchinfo~=1.8.0
```

## Installation

### Clone or Download the Repository
Begin by cloning or downloading this repository to your local machine.

### Update Configuration
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

#### PROJECT_ROOT
- Update this path to the directory where the Python scripts and JSON files of the project are located. This directory will be used as the central repository for essential files.
#### DATASET_ROOT: 
- Modify this path to point to the directory where your datasets are stored. This folder should contain all datasets necessary for the project.
#### STORAGE_ROOT: 
- Adjust this path to the location where you want to save project outputs and other data generated during the execution of the project.

### Create Necessary Folders
Run the __data_paths.py__ script. This will create all the required folders based on the paths specified in the configuration.

### Download Datasets
Obtain the necessary datasets and place them into the DATASET_ROOT directory as specified in your updated configuration

## Usage
### Setting Up Configuration Files
Before running the Python scripts, you need to configure your settings by preparing the following JSON and Python files:
- Configuration for the FCNN (FCNN_config.json)
- Configuration for the HELM (HELM_config.json)
- Configuration for the IPMPDRNN (IPMPDRNN_config.json)
- Configuration for the MPDRNN (MPDRNN_config.json)


Once your configuration files are set up, run the Python scripts to train, test, or augment data

### Workflow
- First of all, it is advisable to run hyperparameter tuning, although config files contain the best settings.
  - There is a separate file for hyperparameter tuning for all available networks. 
- After tuning, you may execute the training and evaluation for the desired network.