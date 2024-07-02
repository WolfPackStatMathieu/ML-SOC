"""
Module for training, evaluating, and visualizing a Support Vector Machine (SVM) classifier.

This module provides functions to train a Support Vector Machine classifier, evaluate its performance on a test set,
and visualize the results using a confusion matrix. It includes the following functions:

- train_svm: Trains the SVM model.
- evaluate_svm_model: Evaluates the trained model and prints various performance metrics.
- plot_confusion_matrix_svm: Plots the confusion matrix for the model predictions.
- model_svm: High-level function to train, evaluate, and plot the SVM model.

Example usage:
--------------
import pandas as pd
from sklearn.model_selection import train_test_split

# Assume X and y are your feature matrix and target vector respectively
X, y = ...  # Load your data here

# Split the dataset into training and testing sets
x_tr, x_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=0)

# Train, evaluate, and plot the SVM model
model_params = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}
model_svm(x_tr, y_tr, x_ts, y_ts, model_params)
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
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


def train_svm(x_train, y_train, **kwargs):
    """
    Train a Support Vector Machine classifier on the provided training data.

    Parameters:
    x_train (DataFrame): Features for training.
    y_train (Series): Target variable for training.
    **kwargs: Additional keyword arguments for the SVC classifier.

    Returns:
    SVC: Trained Support Vector Machine model.
    """
    svc_model = SVC(**kwargs)
    print("Computing....")
    svc_model.fit(x_train, y_train)
    print("Done!")
    return svc_model


def evaluate_svm_model(model, x_test, y_test):
    """
    Evaluate the trained SVM model on the test data and print performance metrics.

    Parameters:
    model (SVC): Trained Support Vector Machine model.
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
    error_svc = (predictions != y_test).mean()
    print(f"Test error: {error_svc:.1%}")

    # Afficher le rapport de classification
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal (class 0)", "Anomalous (class 1)"],
        )
    )

    return predictions


def plot_confusion_matrix_svm(
    y_test, predictions, labels=None, filename="confusion_matrix_svm.png"
):
    """
    Plot the confusion matrix for the SVM model predictions and save it to a file.

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
    plt.title("Support Vector Machine")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(filename)
    plt.close()


def model_svm(x_train, y_train, x_test, y_test, model_params):
    """
    Train, evaluate, and plot the SVM model.

    This function trains a Support Vector Machine classifier on the training data, evaluates it on the test data,
    and plots the confusion matrix.

    Parameters:
    x_train (DataFrame): Features for training.
    y_train (Series): Target variable for training.
    x_test (DataFrame): Features for testing.
    y_test (Series): Target variable for testing.
    model_params (dict): Dictionary containing model parameters (e.g., C, kernel, gamma).
    """
    start_time = time.time()
    model = train_svm(x_train, y_train, **model_params)
    end_time = time.time()
    print(
        f"SUPPORT VECTOR MACHINE (SVM) Execution time: {end_time - start_time:.2f} seconds"
    )
    predictions = evaluate_svm_model(model, x_test, y_test)
    plot_confusion_matrix_svm(y_test, predictions)


# Exemple d'appel de la fonction avec les ensembles d'entraînement et de test
# model_params = {
#     'C': 1.0,
#     'kernel': 'rbf',
#     'gamma': 'scale'
# }
# model_svm(x_tr, y_tr, x_ts, y_ts, model_params)
