# Handwritten Digit Recognition Using ANN

## Project Overview

This project implements a handwritten digit recognition system using an Artificial Neural Network (ANN). It uses the MNIST dataset to classify grayscale images of handwritten digits from **0 to 9**.

The project includes a Jupyter Notebook for data preparation, model training and evaluation, a saved Keras model, and a Streamlit application for trying digit predictions through a user interface.

## Project Details

- **Dataset:** MNIST
- **Total images:** 70,000
- **Training images:** 60,000
- **Testing images:** 10,000
- **Image size:** 28 × 28 pixels
- **Number of classes:** 10 (digits 0–9)
- **Model:** Fully connected Artificial Neural Network
- **Framework:** TensorFlow / Keras

## ANN Architecture

1. Input image: 28 × 28 pixels
2. Flatten layer
3. Dense layer: 128 neurons, ReLU activation
4. Dense layer: 64 neurons, ReLU activation
5. Output layer: 10 neurons, Softmax activation

The model has **109,386 trainable parameters**.

## Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss function | Sparse Categorical Crossentropy |
| Evaluation metric | Accuracy |
| Epochs | 10 |
| Batch size | 32 |
| Training samples | 54,000 |
| Validation samples | 6,000 |
| Test samples | 10,000 |

## Results

| Metric | Result |
|---|---:|
| Final training accuracy | 99.41% |
| Final validation accuracy | 97.60% |
| Test accuracy | 98.03% |
| Test loss | 0.0813 |
| Correct test predictions | 9,803 |
| Incorrect test predictions | 197 |

**Evaluation note:** The 98.03% accuracy is the result on the MNIST test dataset. The Streamlit drawing interface may produce different results on user-drawn digits. In informal testing, the digit 7 was sometimes confused with 2 or 3.

## Folder Structure

```text
Handwritten_Digit_Recognition_ANN/
│
├── Source_Code/
│   ├── Handwritten_Digit_Recognition_ANN.ipynb
│   ├── app.py
│   └── requirements.txt
│
├── Model/
│   └── handwritten_digit_ann.keras
│
├── Dataset/
│   └── README.txt
│
├── Screenshots/
│   ├── dataset_samples.png
│   ├── training_output.png
│   ├── accuracy_loss_graphs.png
│   ├── confusion_matrix.png
│   └── streamlit_app.png
│
├── Results/
│   ├── ANN_Classification_Report.csv
│   └── ANN_Confusion_Matrix.csv
│
└── README.md
```

## Requirements

Python and the libraries used by the project must be installed. The `requirements.txt` file in `02_Source_Code` should contain the packages required by the notebook and Streamlit application.

## How to Run

Open a terminal in the main project folder.

### 1. Install dependencies

```bash
pip install -r 02_Source_Code/requirements.txt
```

### 2. Start the Streamlit application

```bash
streamlit run 02_Source_Code/app.py
```

Make sure the application loads the saved model from:

```text
03_Model/handwritten_digit_ann.keras
```

The model path used in `app.py` should be relative to the main project folder if the application is launched using the command above.

## How to Train the Model

1. Open `02_Source_Code/Handwritten_Digit_Recognition_ANN.ipynb` in Jupyter Notebook or Google Colab.
2. Run the notebook cells in order.
3. The notebook loads the MNIST dataset and performs preprocessing.
4. The ANN is trained and evaluated.
5. Save or download the trained model and result files when the notebook generates them.

## Screenshots and Results

The `05_Screenshots` folder contains screenshots of the dataset, training output, accuracy and loss graphs, confusion matrix, and Streamlit application.

The `06_Results` folder contains the classification report and confusion matrix exported as CSV files.

## Limitations and Future Improvements

- The model is trained on the MNIST dataset and may not generalize equally well to all handwriting styles.
- The fully connected ANN does not preserve spatial relationships between image pixels as effectively as a convolutional neural network.
- User-drawn digits can be affected by stroke thickness, position, size, and preprocessing.
- Future improvements may include using a CNN, data augmentation, and improved preprocessing for user-drawn images.
