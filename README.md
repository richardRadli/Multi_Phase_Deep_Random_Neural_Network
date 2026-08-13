# Multi-Phase Deep Random Neural Network (MPDRNN) System

This project is a microservice-based, Dockerized framework designed for training, testing, evaluating, and **hyperparameter tuning** across various artificial neural network architectures (**FCNN, HELM, DRNN, IPMPDRNN**), complemented by a modern React-based web user interface.

---

## 🔥 Key System Capabilities

* **3-Way Dataset Splitting (Train / Validation / Test):** Dynamically partitions datasets into Training, Validation, and Testing sets. Validation data is strictly reserved for hyperparameter tuning and model selection, ensuring unbiased final evaluation on the Test set.
* **Dual-Backend Hyperparameter Tuning:** All neural network services support automated hyperparameter search powered by both **Optuna** (TPE Sampler, Median Pruner) and **Ray Tune** (ASHA Scheduler).
* **Real-time Task Tracking & Process Control:** Live progress monitoring and instant task abort capabilities controlled via **Celery** workers and **Redis** status signaling.
* **Automatic Dataset Management:** Zero-setup dataset initialization. Missing datasets are automatically downloaded and extracted into a local relative `./datasets` directory upon system launch.

---

## 🏛️ System Architecture

The framework consists of independent microservices communicating seamlessly via Docker Compose:

* **React Frontend (`:5173`)**: Modern web dashboard for configuring dataset splits, running training/tuning tasks, controlling active jobs, and visualizing performance metrics.
* **Dataset Operations Service (`:8000`)**: Handles automatic dataset downloading, archiving, and dynamic **3-way dataset splitting (Train / Validation / Test)**.
* **FCNN Service (`:8001`)**: Manages training, evaluation, and **Optuna / Ray Tune** hyperparameter search for Fully Connected Neural Networks (MLP).
* **HELM Service (`:8002`)**: Handles training, evaluation, and **Optuna / Ray Tune** hyperparameter search for Hierarchical Extreme Learning Machines.
* **DEV-DRNN Service (`:8003`)**: Manages training, evaluation, and **Optuna / Ray Tune** hyperparameter tuning for Deep Random Neural Networks.
* **DEV-DRNN-AUX Service (`:8004`)**: Handles auxiliary DRNN algorithms, multi-phase network pruning (**IPMPDRNN**), and **Optuna / Ray Tune** hyperparameter optimization for pruning thresholds and sub-networks.
* **Redis Broker (`:6379`)**: Asynchronous task queue broker and status store for Celery workers and abort signals.

---

## 🌐 User Interface Workspaces

The React application contains five specialized workspaces:

### 1. 📊 Dataset Workspace (`DatasetWorkspace.tsx`)
* **Purpose:** Dataset management and pre-processing.
* **What it does:** Displays available benchmark datasets (e.g., `mnist`, `connect4`, `usps`, etc.) and allows dynamic **3-way splitting** into Training, Validation, and Testing subsets (e.g., 70% Train / 15% Validation / 15% Test).

### 2. 🧠 FCNN Workspace (`FcnnWorkspace.tsx`)
* **Purpose:** Multi-Layer Perceptron (FCNN) model workbench.
* **What it does:** Supports full training runs, validation-driven testing, and automated hyperparameter optimization using either **Optuna** or **Ray Tune**.

### 3. ⚡ HELM Workspace (`HelmWorkspace.tsx`)
* **Purpose:** Hierarchical Extreme Learning Machine workbench.
* **What it does:** Runs fast, randomized hierarchical network training, validation-guided tuning (**Optuna / Ray Tune**), and final test set evaluations.

### 4. 🎲 Dev DRNN Workspace (`DevDrnnWorkspace.tsx`)
* **Purpose:** Deep Random Neural Network experimental environment.
* **What it does:** Provides configuration, training, validation, and full **Optuna / Ray Tune** hyperparameter search for complex, multi-phase random neural networks.

### 5. ✂️ Dev DRNN Aux / Pruning Workspace (`DevDrnnAuxWorkspace.tsx`)
* **Purpose:** Network pruning (**IPMPDRNN**) and auxiliary network workbench.
* **What it does:** Performs multi-phase network pruning, auxiliary network insertion, and validation-based hyperparameter tuning (**Optuna / Ray Tune**) for pruning ratios and layer dimensions.

---

## 🚀 Getting Started

### Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed on your system.

> ⚠️ **Windows Note:**  
> If Docker throws a file access or volume permission error on the first startup, enable File Sharing in Docker Desktop:  
> Go to **Settings ➔ Resources ➔ File sharing** and ensure `C:\Users` is added to the shared paths.

### Running the System (Single Command)

1. Clone the repository and switch to the target branch:
   ```bash
   git clone <REPO_URL>
   cd Multi_Phase_Deep_Random_Neural_Network
   git checkout <BRANCH_NAME>
   ```

2. Launch all microservices using Docker Compose:
   ```bash
   docker compose up --build
   ```

3. **Automatic Dataset Setup:**  
   On first boot, `dataset-service` checks the local environment. If datasets are missing, it automatically downloads and extracts them into the local `./datasets` directory.

4. Open the web interface in your browser:  
   👉 **`http://localhost:5173`**