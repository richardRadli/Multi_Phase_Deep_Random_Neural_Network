# Deep Randomized Networks for Fast Learning

Deep learning neural networks show a significant improvement over shallow ones in complex problems. Their main disadvantage is their memory requirements, the vanishing gradient problem, and the time consuming solutions to find the best achievable weights and other parameters. Since many applications (such as continuous learning) would need fast training, one possible solution is the application of sub-networks which can be trained very fast. Randomized single layer networks became very popular due to their fast optimization while their extensions, for more complex structures, could increase their prediction accuracy. In our paper we show a new approach to build deep neural models for classification tasks with an iterative, pseudo-inverse optimization technique. We compare the performance with a state-of-the-art backpropagation method and the best known randomized approach called hierarchical extreme learning machine. Computation time and prediction accuracy are evaluated on 12 benchmark datasets, showing that our approach is competitive in many cases.

 The overview of our approach can be drawn by two kinds of phases in four main steps:

-  Create a single ELM network and compute its output weights as the first iteration. This is the first approximation for the solution.
-  Add new neurons to extend the output layer (which now becomes a hidden layer) of the previous phase and add new output neurons at the top level becoming the new output layer.
-  Compute the weights for the new output layer with the matrix inversion technique.
-  Repeat Steps 2-3 as further phases.

<figure align="center">
  <figcaption>Phase 1</figcaption>
  <img src="poc_images/plot_nn_first_phase.svg" alt="phase_1" width="400"/>
</figure>

<figure align="center">
  <figcaption>Phase 2</figcaption>
  <img src="poc_images/plot_nn_second_phase.svg" alt="phase_2" width="400"/>
</figure>

## Datasets
All of the datasets can be accessed in the following link:
https://drive.google.com/file/d/13CnoFZAJj12jia6r3L17YJJ1rANAwSR7/view?usp=sharing
To convert the .txt files into .npy file format, use the convert_dataset.py script.


## Requirement
Make sure you have the following dependencies installed:

```bash
colorlog==6.7.0
keras==2.10.0
matplotlib==3.7.1
numpy==1.23.5
pandas==2.0.0
scipy==1.10.1
seaborn==0.12.2
sklearn==1.2.2
torch==2.0.0+cu117
torchsummary==1.5.1
torchvision==0.15.1+cu117
tqdm==4.65.0
```

## Installation
First, clone/download this repository. In the const.py file you will find this:

```python
root_mapping = {
    'ricsi': {
        "PROJECT_ROOT":
            "D:/research/ELM/storage",
        "DATASET_ROOT":
            "D:/research/ELM/datasets"
    }
}
```

- Update the designated username ('ricsi') to reflect the username associated with your logged-in operating system.
- Utilize PROJECT_ROOT as the central repository for storing essential data such as weights and images
- Employ DATASET_ROOT as the designated directory for managing datasets integral to the functioning of the project.
- const.py will create all the necessary folders.
- Download the datasets and place them into the appropriate folder (in my case it is the DATASET_ROOT.


## Usage
In the config.py file, key parameters and settings crucial for the training, testing are stored. 
These configurations provide a streamlined and organized approach to manage various aspects of the project, ensuring 
adaptability and ease of customization.

## Fully Connectedn Neural Network - FCNN
To train FCNN, set dataset_name in the config.py file within the DatasetConfig() class. Settings regarding the FCNN 
itself can be found in the FCNNConfig() class. FCNN is separated to train and evaluation, so first run train_fcnn.py, 
after training, run eval_fcnn.py

train_fcnn.py logs the loss function with tensorboard, also only the best weights will be saved. The model will be 
trained until early stopping is activated. By default, the patience is 15 epochs. 

<figure align="center">
  <figcaption>Training of a model with FCNN</figcaption>
  <img src="poc_images/fcnn_train.png" alt="training_fcnn" width="600"/>
</figure>

eval_fcnn.py will find the latest directory with the latest model (.pt) file, and will load these weights into the model.
It calculates both the train and the test accuracies, and finally saves the results into a .txt file.

<figure align="center">
  <figcaption>Evaluation of a model with FCNN</figcaption>
  <img src="poc_images/fcnn_eval.png" alt="eval_fcnn" width="600"/>
</figure>


## Hierarchical ELM - HELM

Python implementation of the network, it was originally implemented in MATLAB, the source files of the orignal code can
be found here: https://www.extreme-learning-machines.org/

HELM is not separated to train and evaluation files, both operations happen in the same file. Select the desired dataset
in the config.py file, similarly as with FCNN, and set the parameters of HELM in the _HELMConfig()_ class.

<figure align="center">
  <figcaption>HELM training and testing</figcaption>
  <img src="poc_images/helm.png" alt="helm" width="450"/>
</figure>
