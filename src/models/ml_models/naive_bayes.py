"""
Module for training, evaluating, and visualizing a Naïve Bayes classifier.

This module provides functions to train a Naïve Bayes classifier, evaluate its performance on
a test set,
and visualize the results using a confusion matrix. It includes the following functions:

- train_naive_bayes: Trains the Naïve Bayes model.
- evaluate_naive_bayes_model: Evaluates the trained model and prints various performance metrics.
- plot_confusion_matrix_nb: Plots the confusion matrix for the model predictions.
- model_naive_bayes: High-level function to train, evaluate, and plot the Naïve Bayes model.

Example usage:
--------------
import pandas as pd
from sklearn.model_selection import train_test_split

# Assume X and y are your feature matrix and target vector respectively
X, y = ...  # Load your data here

# Split the dataset into training and testing sets
x_tr, x_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=0)

# Train, evaluate, and plot the Naïve Bayes model
model_naive_bayes(x_tr, y_tr, x_ts, y_ts)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


def train_naive_bayes(x_train, y_train):
    """
    Train a Naïve Bayes classifier on the provided training data.

    Parameters:
    x_train (DataFrame): Features for training.
    y_train (Series): Target variable for training.

    Returns:
    GaussianNB: Trained Naïve Bayes model.
    """
    nb_model = GaussianNB()
    print("Computing....")
    nb_model.fit(x_train, y_train)
    print("Done!")
    return nb_model


def evaluate_naive_bayes_model(model, x_test, y_test):
    """
    Evaluate the trained Naïve Bayes model on the test data and print performance metrics.

    Parameters:
    model (GaussianNB): Trained Naïve Bayes model.
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
    error_nb = (predictions != y_test).mean()
    print(f"Test error: {error_nb:.1%}")

    # Afficher le rapport de classification
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal (class 0)", "Anomalous (class 1)"],
        )
    )

    return predictions


def plot_confusion_matrix_nb(
    y_test, predictions, labels=None, filename="confusion_matrix_nb.png"
):
    """
    Plot the confusion matrix for the Naïve Bayes model predictions and save it to a file.

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
    plt.title("Naïve Bayes")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(filename)
    plt.close()


def model_naive_bayes(x_train, y_train, x_test, y_test):
    """
    Train, evaluate, and plot the Naïve Bayes model.

    This function trains a Naïve Bayes classifier on the training data, evaluates it
    on the test data,
    and plots the confusion matrix.

    Parameters:
    x_train (DataFrame): Features for training.
    y_train (Series): Target variable for training.
    x_test (DataFrame): Features for testing.
    y_test (Series): Target variable for testing.
    """
    model = train_naive_bayes(x_train, y_train)
    predictions = evaluate_naive_bayes_model(model, x_test, y_test)
    plot_confusion_matrix_nb(y_test, predictions)


# Exemple d'appel de la fonction avec les ensembles d'entraînement et de test
# model_naive_bayes(x_tr, y_tr, x_ts, y_ts)
