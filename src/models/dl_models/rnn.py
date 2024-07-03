"""
Module for training, evaluating, and visualizing a Recurrent Neural Network (RNN) model.

This module provides functions to train a Recurrent Neural Network classifier,
evaluate its performance on a test set, and visualize the results using a confusion matrix.
It includes the following functions:

- encode_labels: Encodes labels for binary classification.
- build_rnn_model: Builds a Keras RNN model.
- train_rnn: Trains the RNN model.
- evaluate_rnn_model: Evaluates the trained model and prints various performance metrics.
- plot_confusion_matrix_rnn: Plots the confusion matrix for the model predictions.
- model_rnn: High-level function to train, evaluate, and plot the RNN model.

Example usage:
--------------
import pandas as pd
from sklearn.model_selection import train_test_split

# Assume X and y are your feature matrix and target vector respectively
X, y = ...  # Load your data here

# Split the dataset into training and testing sets
x_tr, x_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=0)

# Train, evaluate, and plot the RNN model
model_rnn(x_tr, y_tr, x_ts, y_ts, rnn_params)
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,  # accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input
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


def build_rnn_model(input_shape, rnn_params):
    """
    Build a Keras RNN model.

    Parameters:
    input_shape (tuple): Shape of the input data.
    rnn_params (dict): Dictionary with parameters for building the RNN model.

    Returns:
    Sequential: Compiled Keras RNN model.
    """
    units = rnn_params["units"]
    activation = rnn_params["activation"]
    learning_rate = rnn_params["learning_rate"]
    optimizer_name = rnn_params["optimizer"]
    loss = rnn_params["loss"]
    metrics = rnn_params["metrics"]

    optimizer_classes = {
        "Adam": Adam,
        "SGD": SGD,
        "RMSprop": RMSprop
    }

    optimizer = optimizer_classes[optimizer_name](learning_rate=learning_rate)

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SimpleRNN(units, activation=activation))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def train_rnn(x_train, y_train, rnn_params):
    """
    Train a Recurrent Neural Network (RNN) classifier.

    Parameters:
    x_train (DataFrame): Features for training.
    y_train (Series): Encoded target variable for training.
    rnn_params (dict): Dictionary with parameters for building and training the RNN model.

    Returns:
    Sequential: Trained RNN model.
    """
    input_shape = (x_train.shape[1], 1)
    model = build_rnn_model(input_shape, rnn_params)
    model.fit(
        x_train,
        y_train,
        epochs=rnn_params["epochs"],
        batch_size=rnn_params["batch_size"],
        verbose=1,
    )
    return model


def evaluate_rnn_model(model, x_test, y_test):
    """
    Evaluate the trained RNN model on the test data and print performance metrics.

    Parameters:
    model (Sequential): Trained RNN model.
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
    error_rnn = (predictions != y_test).mean()
    print(f"Test error: {error_rnn:.1%}")

    # Afficher le rapport de classification
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal (class 0)", "Anomalous (class 1)"],
        )
    )

    return predictions


def plot_confusion_matrix_rnn(
    y_test, predictions, labels=None, filename="confusion_matrix_rnn.png"
):
    """
    Plot the confusion matrix for the RNN model predictions and save it to a file.

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
    plt.title("Recurrent Neural Network")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    filename = "output/fig/" + filename
    plt.savefig(filename)
    plt.close()


def model_rnn(x_train, y_train, x_test, y_test, rnn_params):
    """
    Train, evaluate, and plot the RNN model.

    This function trains a Recurrent Neural Network classifier on the training data,
    evaluates it on the test data, and plots the confusion matrix.

    Parameters:
    x_train (DataFrame): Features for training.
    y_train (Series): Target variable for training.
    x_test (DataFrame): Features for testing.
    y_test (Series): Target variable for testing.
    rnn_params (dict): Dictionary with parameters for building and training the RNN model.
    """
    start_time = time.time()
    y_train_encoded, y_test_encoded = encode_labels(y_train, y_test)
    x_train_reshaped = np.expand_dims(x_train, axis=2)
    x_test_reshaped = np.expand_dims(x_test, axis=2)

    model = train_rnn(x_train_reshaped, y_train_encoded, rnn_params)
    end_time = time.time()
    print(
        f"RECURRENT NEURAL NETWORK Execution time: {end_time - start_time:.2f} seconds"
    )
    predictions = evaluate_rnn_model(model, x_test_reshaped, y_test_encoded)
    plot_confusion_matrix_rnn(y_test_encoded, predictions)


# Exemple d'appel de la fonction avec les ensembles d'entraînement et de test
# rnn_params = {
#     'units': 50,
#     'activation': 'relu',
#     'optimizer': 'Adam',
#     'learning_rate': 0.001,
#     'loss': 'binary_crossentropy',
#     'metrics': ['accuracy'],
#     'epochs': 50,
#     'batch_size': 32
# }
# model_rnn(x_tr, y_tr, x_ts, y_ts, rnn_params)
