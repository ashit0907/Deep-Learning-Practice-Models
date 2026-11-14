# Handwritten Digit Recognition using ANN

A beginner-friendly deep learning project that uses **TensorFlow and Keras** to build an Artificial Neural Network (ANN) capable of recognizing handwritten digits (0–9) using the MNIST dataset.

## Project Overview

The model takes a 28 × 28 grayscale image of a handwritten digit and predicts which digit it represents.

### Dataset

This project uses the **MNIST handwritten digits dataset**, loaded directly through `tf.keras.datasets.mnist`.

- Training images: 60,000
- Testing images: 10,000
- Image dimensions: 28 × 28 pixels
- Number of classes: 10 (digits 0–9)

The dataset is downloaded automatically when the notebook is executed for the first time.

## Model Architecture

The project uses a Sequential Artificial Neural Network with the following architecture:

| Layer | Configuration |
|---|---|
| Input | 28 × 28 |
| Flatten | Converts 2D images into 1D vectors |
| Dense | 128 neurons, ReLU |
| Dense | 32 neurons, ReLU |
| Output | 10 neurons, Softmax |

### Model Configuration

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Evaluation Metric: Accuracy
- Epochs: 25
- Batch Size: 32
- Validation Split: 20%

## Project Workflow

1. Import required libraries.
2. Load the MNIST dataset.
3. Explore and visualize sample images.
4. Normalize pixel values between 0 and 1.
5. Build the ANN architecture.
6. Compile and train the model.
7. Evaluate performance on the test dataset.
8. Generate a classification report.
9. Visualize the confusion matrix.
10. Plot training and validation accuracy and loss.
11. Test predictions on individual handwritten digits.
12. Save and reload the trained model.

## Project Structure

```text
Handwritten-Digit-Recognition/
│
├── Handwritten_digit_pred_ANN.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
└── models/
    └── handwritten_digit_ann.keras
```

The `models/` directory and trained model file are generated automatically when the corresponding notebook cells are executed.

## Requirements

- Python 3.10 or 3.11 recommended
- TensorFlow
- Matplotlib
- Scikit-learn
- Jupyter Notebook

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone this repository:

```bash
git clone <repository-url>
```

2. Navigate to the project directory:

```bash
cd Handwritten-Digit-Recognition
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Launch Jupyter Notebook:

```bash
jupyter notebook
```

5. Open `Handwritten_digit_pred_ANN.ipynb` and execute the cells sequentially.

The MNIST dataset will be downloaded automatically, and the trained model will be saved in the `models/` directory.

## Model Evaluation

The notebook evaluates the trained model using:

- Test accuracy and loss
- Classification report (Precision, Recall and F1-score)
- Confusion matrix
- Training and validation accuracy curves
- Training and validation loss curves

## Future Improvements

- Experiment with different ANN architectures and hyperparameters.
- Build a CNN-based handwritten digit recognition model.
- Develop an interface for predicting digits using uploaded images.
- Compare ANN and CNN performance.

---