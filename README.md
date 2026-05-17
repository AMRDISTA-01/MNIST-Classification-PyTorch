# MNIST Digit Classification: ANN vs. Deep Learning

This repository contains a complete pipeline for classifying handwritten digits from the famous MNIST dataset using PyTorch. This project was developed to fulfill the requirements of two separate courses: **Artificial Neural Networks (ANN)** and **Deep Learning (DL)**.

## Project Overview
The main objective of this project is to build, train, evaluate, and compare two different neural network architectures to solve the same image classification problem:
1. **Multilayer Perceptron (MLP):** Fulfilling the ANN course requirement.
2. **Convolutional Neural Network (CNN):** Fulfilling the Deep Learning course requirement.

## Features & Techniques Used
- **Framework:** Built entirely using `PyTorch`.
- **Data Augmentation:** Applied random rotations to the training set to improve model generalization.
- **Regularization:** Used `nn.Dropout` in both models to prevent overfitting.
- **Optimization:** Utilized the `Adam` optimizer and `CrossEntropyLoss`.
- **Early Stopping:** Implemented custom early stopping with `copy.deepcopy` to save and restore the best model weights based on validation loss.
- **Evaluation & Visualization:** Generated Loss/Accuracy learning curves and Confusion Matrices to deeply analyze model performance.

## Models Architecture
### 1. MLP Model (ANN Requirement)
- Flattens the 28x28 input image into a 1D vector of 784 features.
- Contains hidden `Linear` (Dense) layers with `ReLU` activation functions.
- Includes `Dropout` layers for regularization.

### 2. CNN Model (Deep Learning Requirement)
- Keeps the spatial structure of the 2D images.
- Uses `Conv2d` layers for automatic feature extraction (edges, curves).
- Uses `MaxPool2d` for downsampling and reducing computational load.
- Flattens the final feature maps and passes them through a fully connected (`Linear`) output layer.

## Results & Conclusion
Both models successfully learned to classify the digits. However, the **CNN significantly outperformed the MLP**.
- **MLP Performance:** Showed good accuracy but struggled slightly with visually similar digits (e.g., confusing 4 with 9, or 7 with 2) because flattening the image destroys spatial relationships.
- **CNN Performance:** Achieved near-perfect accuracy on the test set. The confusion matrix for the CNN is remarkably clean, proving its superiority in extracting 2D spatial features and handling computer vision tasks.

## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/Your-Repo-Name.git](https://github.com/YourUsername/Your-Repo-Name.git)
