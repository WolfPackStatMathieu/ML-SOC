"""
Module for training, evaluating, and visualizing a Random
Forest classifier.

This module provides functions to train a Random
Forest classifier, evaluate its performance on a test
set, and visualize the results using a confusion matrix. 
It includes the following functions:

- train_random_forest: Trains the Random Forest model.
- evaluate_model: Evaluates the trained model and prints
various performance metrics.
- plot_confusion_matrix: Plots the confusion matrix for the
model predictions.
- model_random_forest: High-level function to train,
evaluate, and plot the Random Forest model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix


def train_random_forest(x_train, y_train):
    """
    Train a Random Forest classifier on the provided training data.

    Parameters:
    x_train (DataFrame): Features for training.
    y_train (Series): Target variable for training.

    Returns:
    RandomForestClassifier: Trained Random Forest model.
    """
    random_forest_model = RandomForestClassifier(random_state=1000)
    print('Computing....')
    random_forest_model.fit(x_train, y_train)
    print('Done!')
    return random_forest_model



def evaluate_model(model, x_test, y_test):
    """
    Evaluate the trained Random Forest model on the test data and print performance metrics.

    Parameters:
    model (RandomForestClassifier): Trained Random Forest model.
    x_test (DataFrame): Features for testing.
    y_test (Series): Target variable for testing.

    Returns:
    ndarray: Predictions made by the model on the test data.
    """
    predictions = model.predict(x_test)

    # Évaluer le modèle
    print('MAE:', mean_absolute_error(y_test, predictions))
    print('Accuracy:', accuracy_score(y_test, predictions))
    print('Precision:', precision_score(y_test, predictions, average='weighted', labels=np.unique(predictions)))
    print('Recall:', recall_score(y_test, predictions, average='weighted', labels=np.unique(predictions)))
    print('F1:', f1_score(y_test, predictions, average='weighted', labels=np.unique(predictions)))
    print('ROC AUC:', roc_auc_score(y_test, predictions, average='weighted', labels=np.unique(predictions)))
    error_rt = (predictions != y_test).mean()
    print("Test error: {:.1%}".format(error_rt))

    # Afficher le rapport de classification
    print(classification_report(y_test, predictions, target_names=['Normal (class 0)', 'Anomalous (class 1)']))
    
    return predictions


def plot_confusion_matrix(y_test, predictions, labels=['Normal', 'Anomalous']):
    """
    Plot the confusion matrix for the model predictions.

    Parameters:
    y_test (Series): True labels for the test data.
    predictions (ndarray): Predicted labels by the model.
    labels (list): List of label names for the confusion matrix.
    """
    cm = confusion_matrix(y_test, predictions)
    cm = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='', xticklabels=labels, yticklabels=labels)
    plt.title("Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def model_random_forest(x_train, y_train, x_test, y_test):
    """
    Train, evaluate, and plot the Random Forest model.

    This function trains a Random Forest classifier on the training data, evaluates it on the test data, and plots the confusion matrix.

    Parameters:
    x_train (DataFrame): Features for training.
    y_train (Series): Target variable for training.
    x_test (DataFrame): Features for testing.
    y_test (Series): Target variable for testing.
    """
    model = train_random_forest(x_train, y_train)
    predictions = evaluate_model(model, x_test, y_test)
    plot_confusion_matrix(y_test, predictions)

# Exemple d'appel de la fonction avec les ensembles d'entraînement et de test
# model_random_forest(x_tr, y_tr, x_ts, y_ts)
