# MNIST Digit Classification

This project demonstrates how to build a simple neural network for classifying handwritten digits from the MNIST dataset using Keras and TensorFlow. The model is trained to recognize digits from 0 to 9 and is evaluated for accuracy on a test set.

## How to Use the `.ipynb` File

1. **Clone the Repository**: Download or clone this repository to your local machine using:
    ```bash
    git clone git@github.com:HeisenbergHK/MNIST-Digit-Classification.git
    ```
   
2. **Open in Google Colab**: 
   - You can upload the `.ipynb` file directly to [Google Colab](https://colab.research.google.com/) for easy execution.
   - Alternatively, you can run it on your local machine using Jupyter Notebook.

3. **Run the Notebook**: Simply execute the cells sequentially. The notebook:
    - Loads and visualizes the MNIST dataset.
    - Prepares the data for training.
    - Defines and trains a neural network model.
    - Evaluates the performance and visualizes errors.

## Code Overview

### 1. **Imports**

The necessary libraries like `numpy`, `matplotlib`, `seaborn`, and `keras` are imported. The MNIST dataset is provided by Keras.

### 2. **Data Loading and Visualization**

- The MNIST dataset is loaded and visualized. A sample of each digit (0-9) is displayed.
- The dataset consists of 60,000 training images and 10,000 test images, each of size 28x28 pixels.

### 3. **Data Preparation**

- The labels are one-hot encoded to match the categorical output of the neural network.
- The pixel values of the images are normalized to the range [0, 1] to improve training efficiency.
- The images are reshaped from 28x28 matrices to 784-dimensional vectors to be processed by a fully connected neural network.

### 4. **Model Architecture**

- A fully connected neural network (multi-layer perceptron) is created using Keras’ Sequential API.
- The model has:
  - Two dense hidden layers with 128 neurons each, using ReLU activation.
  - A dropout layer with a 25% dropout rate to reduce overfitting.
  - An output layer with 10 neurons (one for each digit), using softmax activation.

### 5. **Training the Model**

- The model is compiled with the categorical cross-entropy loss function, Adam optimizer, and accuracy metric.
- It is trained for 10 epochs, with a batch size of 512.

### 6. **Model Evaluation**

- After training, the model's accuracy is evaluated on the test set.
- The `argmax` function is used to convert the predicted probabilities into predicted class labels.

![prediction example](https://github.com/user-attachments/assets/42d1ba55-8252-42d2-99cc-6340f29f47b6)

### 7. **Confusion Matrix**

- A confusion matrix is generated to visualize the performance of the model across all classes.
- The matrix is plotted using Seaborn for better readability.

![confusion matrix](https://github.com/user-attachments/assets/7b5434a3-bbd3-4872-b543-27f55390a0ee)

### 8. **Investigating Errors**

- The samples where the model made incorrect predictions are identified.
- The top 10 errors (where the model was most confident in its incorrect prediction) are displayed alongside the true and predicted labels.

![errors](https://github.com/user-attachments/assets/5be1438a-cada-43de-9dba-63bfb424327d)

## Results

The model achieves reasonable accuracy on the test set and provides insights into its performance by visualizing incorrect predictions and the confusion matrix.

## Requirements

- Python 3.x
- Keras
- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
