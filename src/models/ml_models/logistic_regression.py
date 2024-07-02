"""
Module for training, evaluating, and visualizing a Logistic Regression classifier.

This module provides functions to train a Logistic Regression classifier, evaluate its performance
on a test set,
and visualize the results using a confusion matrix. It includes the following functions:

- train_logistic_regression: Trains the Logistic Regression model.
- evaluate_logistic_regression_model: Evaluates the trained model and prints various performance
metrics.
- plot_confusion_matrix_lr: Plots the confusion matrix for the model predictions.
- model_logistic_regression: High-level function to train, evaluate, and plot the Logistic
Regression model.

Example usage:
--------------
import pandas as pd
from sklearn.model_selection import train_test_split

# Assume X and y are your feature matrix and target vector respectively
X, y = ...  # Load your data here

# Split the dataset into training and testing sets
x_tr, x_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=0)

# Train, evaluate, and plot the Logistic Regression model
model_params = {'random_state': 42, 'max_iter': 1000}
model_logistic_regression(x_tr, y_tr, x_ts, y_ts, model_params)
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
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


def train_logistic_regression(x_train, y_train, random_state=42, max_iter=1000):
    """
    Train a Logistic Regression classifier on the provided training data.

    Parameters:
    x_train (DataFrame): Features for training.
    y_train (Series): Target variable for training.
    random_state (int): Random state for reproducibility.
    max_iter (int): Maximum number of iterations.

    Returns:
    LogisticRegression: Trained Logistic Regression model.
    """
    lr_model = LogisticRegression(random_state=random_state, max_iter=max_iter)
    print("Computing....")
    lr_model.fit(x_train, y_train)
    print("Done!")
    return lr_model


def evaluate_logistic_regression_model(model, x_test, y_test):
    """
    Evaluate the trained Logistic Regression model on the test data and print performance metrics.

    Parameters:
    model (LogisticRegression): Trained Logistic Regression model.
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
    error_lr = (predictions != y_test).mean()
    print(f"Test error: {error_lr:.1%}")

    # Afficher le rapport de classification
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal (class 0)", "Anomalous (class 1)"],
        )
    )

    return predictions


def plot_confusion_matrix_lr(
    y_test, predictions, labels=None, filename="confusion_matrix_lr.png"
):
    """
    Plot the confusion matrix for the Logistic Regression model predictions and save it to a file.

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
    plt.title("Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    filename = "output/fig/" + filename
    plt.savefig(filename)
    plt.close()


def model_logistic_regression(x_train, y_train, x_test, y_test, model_params):
    """
    Train, evaluate, and plot the Logistic Regression model.

    This function trains a Logistic Regression classifier on the training data, evaluates it on the
    test data,
    and plots the confusion matrix.

    Parameters:
    x_train (DataFrame): Features for training.
    y_train (Series): Target variable for training.
    x_test (DataFrame): Features for testing.
    y_test (Series): Target variable for testing.
    model_params (dict): Dictionary containing model parameters (e.g., random_state, max_iter).
    """
    start_time = time.time()
    model = train_logistic_regression(x_train, y_train, **model_params)
    end_time = time.time()
    print(f"LOGISTIC REGRESSION Execution time: {end_time - start_time:.2f} seconds")
    predictions = evaluate_logistic_regression_model(model, x_test, y_test)
    plot_confusion_matrix_lr(y_test, predictions)


# Exemple d'appel de la fonction avec les ensembles d'entraînement et de test
# model_params = {
#     'random_state': 42,
#     'max_iter': 1000
# }
# model_logistic_regression(x_tr, y_tr, x_ts, y_ts, model_params)
