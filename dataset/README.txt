DATASET INFORMATION
===================

Project: Handwritten Digit Recognition Using Artificial Neural Network (ANN)

Dataset Name:
MNIST Handwritten Digits Dataset

Dataset Source:
https://www.tensorflow.org/datasets/catalog/mnist

Dataset Description:
The MNIST dataset contains 70,000 grayscale images of handwritten digits from 0 to 9.
Each image has a size of 28 x 28 pixels.

Dataset Split:
- Training images: 60,000
- Testing images: 10,000

Preprocessing Used:
- Images are represented as 28 x 28 grayscale pixel arrays.
- Pixel values are normalized to the range 0 to 1 by dividing by 255.0.
- The original training set is split into:
  - 54,000 training images
  - 6,000 validation images
- The separate 10,000-image test set is used for final evaluation.

Labels:
Each image has a label from 0 to 9, representing the handwritten digit.

Dataset Storage:
The MNIST dataset is loaded by the project notebook using the relevant machine-learning
library. Therefore, a separate copy of the dataset does not need to be stored in this
folder unless required by the project supervisor.

Usage:
Run the notebook in 02_Source_Code to load and preprocess the MNIST dataset.

Note:
The reported test accuracy of 98.03% is measured on the MNIST test set. It should not be
interpreted as a guarantee of the same accuracy on every image drawn by a user in the
Streamlit application.
