"""
Module for training, evaluating, and visualizing a K-Nearest Neighbors (KNN) classifier.

This module provides functions to train a K-Nearest Neighbors classifier, evaluate its performance
on a test set,
and visualize the results using a confusion matrix. It includes the following functions:

- train_knn: Trains the K-Nearest Neighbors model.
- evaluate_knn_model: Evaluates the trained model and prints various performance metrics.
- plot_confusion_matrix_knn: Plots the confusion matrix for the model predictions.
- model_knn: High-level function to train, evaluate, and plot the K-Nearest Neighbors model.

Example usage:
--------------
import pandas as pd
from sklearn.model_selection import train_test_split

# Assume X and y are your feature matrix and target vector respectively
X, y = ...  # Load your data here

# Split the dataset into training and testing sets
x_tr, x_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=0)

# Train, evaluate, and plot the K-Nearest Neighbors model
model_knn(x_tr, y_tr, x_ts, y_ts, n_neighbors=9)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


def train_knn(x_train, y_train, n_neighbors=9):
    """
    Train a K-Nearest Neighbors classifier on the provided training data.

    Parameters:
    x_train (DataFrame): Features for training.
    y_train (Series): Target variable for training.
    n_neighbors (int): Number of neighbors to use.

    Returns:
    KNeighborsClassifier: Trained K-Nearest Neighbors model.
    """
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(x_train, y_train)
    return knn_model


def evaluate_knn_model(model, x_test, y_test):
    """
    Evaluate the trained K-Nearest Neighbors model on the test data and print performance metrics.

    Parameters:
    model (KNeighborsClassifier): Trained K-Nearest Neighbors model.
    x_test (DataFrame): Features for testing.
    y_test (Series): Target variable for testing.

    Returns:
    ndarray: Predictions made by the model on the test data.
    """
    predictions = model.predict(x_test)

    # Évaluer le modèle
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(
        "Precision:",
        precision_score(
            y_test, predictions, average="weighted", labels=np.unique(predictions)
        ),
    )
    print(
        "Recall:",
        recall_score(
            y_test, predictions, average="weighted", labels=np.unique(predictions)
        ),
    )
    print(
        "F1:",
        f1_score(
            y_test, predictions, average="weighted", labels=np.unique(predictions)
        ),
    )
    print(
        "ROC AUC:",
        roc_auc_score(
            y_test, predictions, average="weighted", labels=np.unique(predictions)
        ),
    )
    error_knn = (predictions != y_test).mean()
    print(f"Test error: {error_knn:.1%}")

    # Afficher le rapport de classification
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal (class 0)", "Anomalous (class 1)"],
        )
    )

    return predictions


def plot_confusion_matrix_knn(y_test, predictions, labels=None):
    """
    Plot the confusion matrix for the K-Nearest Neighbors model predictions.

    Parameters:
    y_test (Series): True labels for the test data.
    predictions (ndarray): Predicted labels by the model.
    labels (list): List of label names for the confusion matrix.
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
    plt.title("K-Nearest Neighbors")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def model_knn(x_train, y_train, x_test, y_test, n_neighbors=9):
    """
    Train, evaluate, and plot the K-Nearest Neighbors model.

    This function trains a K-Nearest Neighbors classifier on the training data, evaluates it on
    the test data, and plots the confusion matrix.

    Parameters:
    x_train (DataFrame): Features for training.
    y_train (Series): Target variable for training.
    x_test (DataFrame): Features for testing.
    y_test (Series): Target variable for testing.
    n_neighbors (int): Number of neighbors to use for the K-Nearest Neighbors classifier.
    """
    model = train_knn(x_train, y_train, n_neighbors)
    predictions = evaluate_knn_model(model, x_test, y_test)
    plot_confusion_matrix_knn(y_test, predictions)


# Exemple d'appel de la fonction avec les ensembles d'entraînement et de test
# model_knn(x_tr, y_tr, x_ts, y_ts)
