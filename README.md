# Multi-Phase Deep Random Neural Network (MPDRNN) System

This project is a microservice-based, Dockerized framework designed for training, testing, and evaluating various artificial neural network architectures (**FCNN, HELM, MPDRNN**), complemented by a modern React-based web user interface.

---

## 🔥 Key System Capabilities

* **2-Way Dataset Splitting (Train / Test):** Dynamically partitions datasets into Training and Testing sets based on custom user-defined ratios.
* **Streamlined Training & Evaluation:** Full pipeline for model configuration, training, and evaluation across different network architectures.
* **Real-time Task Tracking:** Live progress monitoring and task management powered by **Celery** workers and **Redis** status store.
* **Automatic Dataset Management:** Zero-setup dataset initialization. Missing datasets are automatically downloaded and extracted into a local relative `./datasets` directory upon system launch.

---

## 🏛️ System Architecture

The framework consists of independent microservices communicating seamlessly via Docker Compose:

* **React Frontend (`:5173`)**: Modern web dashboard for configuring dataset splits, executing training/testing jobs, and visualizing performance metrics.
* **Dataset Operations Service (`:8000`)**: Handles automatic dataset downloading, archiving, and dynamic **2-way dataset splitting (Train / Test)**.
* **FCNN Service (`:8001`)**: Manages training and testing for Fully Connected Neural Networks (MLP).
* **HELM Service (`:8002`)**: Handles training and evaluation for Hierarchical Extreme Learning Machine models.
* **MPDRNN Service (`:8003`)**: Manages training and evaluation for Multi-Phase Deep Random Neural Networks.
* **Redis Broker (`:6379`)**: Asynchronous task queue broker and status store for Celery workers.

---

## 🌐 User Interface Workspaces

The React application contains four specialized workspaces:

### 1. 📊 Dataset Workspace (`DatasetWorkspace.tsx`)
* **Purpose:** Dataset management and pre-processing.
* **What it does:** Displays available benchmark datasets (e.g., `mnist`, `connect4`, `usps`, etc.) and allows dynamic **2-way splitting** into Training and Testing subsets based on a chosen ratio.

### 2. 🧠 FCNN Workspace (`FcnnWorkspace.tsx`)
* **Purpose:** Multi-Layer Perceptron (FCNN) model workbench.
* **What it does:** Supports full training runs, parameter setup, and test set evaluations for FCNN models.

### 3. ⚡ HELM Workspace (`HelmWorkspace.tsx`)
* **Purpose:** Hierarchical Extreme Learning Machine workbench.
* **What it does:** Runs fast, randomized hierarchical network training and final test set evaluations.

### 4. 🎲 MPDRNN Workspace (`MpdrnnWorkspace.tsx`)
* **Purpose:** Multi-Phase Deep Random Neural Network experimental environment.
* **What it does:** Provides configuration, multi-phase training, and testing options for Multi-Phase Deep Random Neural Networks.

---

## 🚀 Getting Started

### Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed on your system.

> ⚠️ **Windows Note:**  
> If Docker throws a file access or volume permission error on the first startup, enable File Sharing in Docker Desktop:  
> Go to **Settings ➔ Resources ➔ File sharing** and ensure `C:\Users` is added to the shared paths.

### Running the System (Single Command)

1. Clone the repository and switch to this branch:
   ```bash
   git clone <REPO_URL>
   cd Multi_Phase_Deep_Random_Neural_Network
   git checkout lion17-internship
   ```

2. Launch all microservices using Docker Compose:
   ```bash
   docker compose up --build
   ```

3. **Automatic Dataset Setup:**  
   On first boot, `dataset-service` checks the local environment. If datasets are missing, it automatically downloads and extracts them into the local `./datasets` directory.

4. Open the web interface in your browser:  
   👉 **`http://localhost:5173`**