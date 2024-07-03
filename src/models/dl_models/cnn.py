"""
Module for training, evaluating, and visualizing a Convolutional Neural Network (CNN) model.

This module provides functions to train a Convolutional Neural Network classifier,
evaluate its performance on a test set, and visualize the results using a confusion matrix.
It includes the following functions:

- encode_labels: Encodes labels for binary classification.
- build_cnn_model: Builds a Keras CNN model.
- train_cnn: Trains the CNN model.
- evaluate_cnn_model: Evaluates the trained model and prints various performance metrics.
- plot_confusion_matrix_cnn: Plots the confusion matrix for the model predictions.
- model_cnn: High-level function to train, evaluate, and plot the CNN model.

Example usage:
--------------
import pandas as pd
from sklearn.model_selection import train_test_split

# Assume X and y are your feature matrix and target vector respectively
X, y = ...  # Load your data here

# Split the dataset into training and testing sets
x_tr, x_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=0)

# Train, evaluate, and plot the CNN model
model_cnn(x_tr, y_tr, x_ts, y_ts, cnn_params)
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


def encode_labels(y_train, y_test):
    """
    Encode labels for binary classification.

    Parameters:
    y_train (Series): Target variable for training.
    y_test (Series): Target variable for testing.

    Returns:
    y_train_encoded, y_test_encoded: Encoded labels for training and testing.
    """
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    return y_train_encoded, y_test_encoded


def build_cnn_model(input_shape, cnn_params):
    """
    Build a Keras CNN model.

    Parameters:
    input_shape (tuple): Shape of the input data.
    cnn_params (dict): Dictionary with parameters for building the CNN model.

    Returns:
    Sequential: Compiled Keras CNN model.
    """
    learning_rate = cnn_params["learning_rate"]
    optimizer_name = cnn_params["optimizer"]
    loss = cnn_params["loss"]
    metrics = cnn_params["metrics"]

    optimizer_classes = {
        "Adam": Adam,
        "SGD": SGD,
        "RMSprop": RMSprop
    }

    optimizer = optimizer_classes[optimizer_name](learning_rate=learning_rate)

    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    
    for layer in cnn_params["layers"][1:]:  # Skip the first layer as it's already added
        layer_name = list(layer.keys())[0]
        layer_params = layer[layer_name]
        model.add(Dense(**layer_params))
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def train_cnn(x_train, y_train, x_test, y_test, cnn_params):
    """
    Train a Convolutional Neural Network (CNN) classifier.

    Parameters:
    x_train (DataFrame): Features for training.
    y_train (Series): Encoded target variable for training.
    x_test (DataFrame): Features for testing.
    y_test (Series): Encoded target variable for testing.
    cnn_params (dict): Dictionary with parameters for building and training the CNN model.

    Returns:
    Sequential: Trained CNN model.
    """
    input_shape = (x_train.shape[1],)
    model = build_cnn_model(input_shape, cnn_params)
    model.fit(
        x_train,
        y_train,
        epochs=cnn_params["epochs"],
        batch_size=cnn_params["batch_size"],
        validation_data=(x_test, y_test),
        verbose=1,
    )
    return model


def evaluate_cnn_model(model, x_test, y_test):
    """
    Evaluate the trained CNN model on the test data and print performance metrics.

    Parameters:
    model (Sequential): Trained CNN model.
    x_test (DataFrame): Features for testing.
    y_test (Series): Encoded target variable for testing.

    Returns:
    ndarray: Predictions made by the model on the test data.
    """
    accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
    predictions = (model.predict(x_test) > 0.5).astype("int32")

    # Évaluer le modèle
    print(f"Accuracy: {accuracy}")
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, average="weighted"))
    print("Recall:", recall_score(y_test, predictions, average="weighted"))
    print("F1:", f1_score(y_test, predictions, average="weighted"))
    print("ROC AUC:", roc_auc_score(y_test, predictions, average="weighted"))
    error_cnn = (predictions != y_test).mean()
    print(f"Test error: {error_cnn:.1%}")

    # Afficher le rapport de classification
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal (class 0)", "Anomalous (class 1)"],
        )
    )

    return predictions


def plot_confusion_matrix_cnn(
    y_test, predictions, labels=None, filename="confusion_matrix_cnn.png"
):
    """
    Plot the confusion matrix for the CNN model predictions and save it to a file.

    Parameters:
    y_test (Series): True labels for the test data.
    predictions (ndarray): Predicted labels by the model.
    labels (list): List of label names for the confusion matrix.
    filename (str): Name of the file to save the plot.
    """
    if labels is None:
        labels = ["Normal", "Anomalous"]

    cm = confusion_matrix(y_test, predictions)
    cm = pd.DataFrame(cm, index=["0", "1"], columns=["0", "1"])

    plt.figure(figsize=(10, 10))
    sns.heatmap(
        cm,
        cmap="Blues",
        linecolor="black",
        linewidth=1,
        annot=True,
        fmt="",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Convolutional Neural Network")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    filename = "output/fig/" + filename
    plt.savefig(filename)
    plt.close()


def model_cnn(x_train, y_train, x_test, y_test, cnn_params):
    """
    Train, evaluate, and plot the CNN model.

    This function trains a Convolutional Neural Network classifier on the training data,
    evaluates it on the test data, and plots the confusion matrix.

    Parameters:
    x_train (DataFrame): Features for training.
    y_train (Series): Target variable for training.
    x_test (DataFrame): Features for testing.
    y_test (Series): Target variable for testing.
    cnn_params (dict): Dictionary with parameters for building and training the CNN model.
    """
    start_time = time.time()
    y_train_encoded, y_test_encoded = encode_labels(y_train, y_test)

    # Convert the training and testing sets to NumPy arrays
    x_train_np = x_train.to_numpy()
    x_test_np = x_test.to_numpy()

    # Reshape training and testing sets for CNN model
    x_train_np = x_train_np.reshape(x_train_np.shape[0], -1)  # Flatten the input to a 1D array
    x_test_np = x_test_np.reshape(x_test_np.shape[0], -1)  # Flatten the input to a 1D array

    model = train_cnn(x_train_np, y_train_encoded, x_test_np, y_test_encoded, cnn_params)
    end_time = time.time()
    print(f"CONVOLUTIONAL NEURAL NETWORK Execution time: {end_time - start_time:.2f} seconds")
    predictions = evaluate_cnn_model(model, x_test_np, y_test_encoded)
    plot_confusion_matrix_cnn(y_test_encoded, predictions)


# Exemple d'appel de la fonction avec les ensembles d'entraînement et de test
# cnn_params = {
#     'layers': [
#         {'Flatten': {'input_shape': (x_tr_np.shape[1],)}},
#         {'Dense': {'units': 64, 'activation': 'relu'}},
#         {'Dense': {'units': 32, 'activation': 'relu'}},
#         {'Dense': {'units': 1, 'activation': 'sigmoid'}}
#     ],
#     'optimizer': 'Adam',
#     'learning_rate': 0.001,
#     'loss': 'binary_crossentropy',
#     'metrics': ['accuracy'],
#     'epochs': 30,
#     'batch_size': 32
# }
# model_cnn(x_tr, y_tr, x_ts, y_ts, cnn_params)
