Cranial Deformity Classification (3D CNN Project)

This project classifies different types of cranial deformities using 3D skull models stored in .obj format. The models are converted into voxel grids and trained using deep learning models like 3D CNN, 3D ResNet, and 3D DenseNet.

Features

Loads and extracts modelsSynth.zip dataset.(please access the dataset from here)

Converts .obj files into 3D voxel grids.

Preprocesses data and prepares labels.

Trains three different models for comparison:

Simple 3D CNN

3D ResNet

3D DenseNet

Provides various visualizations such as voxel slices, class distribution, training accuracy, training loss, and confusion matrices.

Requirements

Install the necessary libraries:

pip install numpy pandas matplotlib seaborn trimesh tensorflow scikit-learn

How It Works

Extract the dataset.

Load all .obj files from each class folder.

Convert each 3D model into a voxel grid of size 32×32×32.

Encode labels and split the data into training and testing sets.

Train the 3D CNN, ResNet, and DenseNet models.

Plot results and evaluate using accuracy and confusion matrices.

Dataset

Place modelsSynth.zip in the working directory.
The expected structure after extraction:

/dataset/modelsSynth
    /Class1
    /Class2
    /Class3
    ...


Each folder contains .obj files.

Training

Run the script to train the models:

create_3d_cnn_model()

create_3d_resnet_model()

DenseNet3D()

Each model is trained for 20 epochs with validation.

Output

The program generates:

Training/validation accuracy and loss graphs

Confusion matrices

Voxel visualizations

PCA plot of voxel features

Class distribution chart
