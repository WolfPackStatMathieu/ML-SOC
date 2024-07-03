"""
Module for training, evaluating, and visualizing a Decision Tree classifier.

This module provides functions to train a Decision Tree classifier, evaluate its performance on a
test set,
and visualize the results using a confusion matrix. It includes the following functions:

- train_decision_tree: Trains the Decision Tree model.
- evaluate_decision_tree_model: Evaluates the trained model and prints various performance metrics.
- plot_confusion_matrix_dt: Plots the confusion matrix for the model predictions.
- model_decision_tree: High-level function to train, evaluate, and plot the Decision Tree model.

Example usage:
--------------
import pandas as pd
from sklearn.model_selection import train_test_split

# Assume X and y are your feature matrix and target vector respectively
X, y = ...  # Load your data here

# Split the dataset into training and testing sets
x_tr, x_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=0)

# Train, evaluate, and plot the Decision Tree model
model_decision_tree(x_tr, y_tr, x_ts, y_ts, random_state=2)
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
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


def train_decision_tree(x_train, y_train, params):
    """
    Train a Decision Tree classifier on the provided training data.

    Parameters:
    x_train (DataFrame): Features for training.
    y_train (Series): Target variable for training.
    params (dict): Hyperparameters for Decision Tree.

    Returns:
    DecisionTreeClassifier: Trained Decision Tree model.
    """
    dt_model = DecisionTreeClassifier(**params)
    print("Computing....")
    dt_model.fit(x_train, y_train)
    print("Done!")
    return dt_model


def evaluate_decision_tree_model(model, x_test, y_test):
    """
    Evaluate the trained Decision Tree model on the test data and print performance metrics.

    Parameters:
    model (DecisionTreeClassifier): Trained Decision Tree model.
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
    error_dt = (predictions != y_test).mean()
    print(f"Test error: {error_dt:.1%}")

    # Afficher le rapport de classification
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal (class 0)", "Anomalous (class 1)"],
        )
    )

    return predictions


def plot_confusion_matrix_dt(
    y_test, predictions, labels=None, filename="confusion_matrix_dt.png"
):
    """
    Plot the confusion matrix for the Decision Tree model predictions.

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
    plt.title("Decision Tree")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    filename = "output/fig/" + filename
    plt.savefig(filename)
    plt.close()


def model_decision_tree(x_train, y_train, x_test, y_test, params):
    """
    Train, evaluate, and plot the Decision Tree model.

    This function trains a Decision Tree classifier on the training data, evaluates it on the
    test data, and plots the confusion matrix.

    Parameters:
    x_train (DataFrame): Features for training.
    y_train (Series): Target variable for training.
    x_test (DataFrame): Features for testing.
    y_test (Series): Target variable for testing.
    params (dict): Hyperparameters for Decision Tree.
    """
    start_time = time.time()
    model = train_decision_tree(x_train, y_train, params)
    end_time = time.time()
    print(f"DECISION TREE Execution time: {end_time - start_time:.2f} seconds")
    predictions = evaluate_decision_tree_model(model, x_test, y_test)
    plot_confusion_matrix_dt(y_test, predictions)

# Exemple d'appel de la fonction avec les ensembles d'entraînement et de test
# model_decision_tree(x_tr, y_tr, x_ts, y_ts, {"random_state": 2})
