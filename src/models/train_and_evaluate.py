"""
Module for training and evaluating machine learning models.

This module provides functions to train and evaluate multiple
machine learning models based on provided parameters.

Example usage:
--------------
train_and_evaluate_models(x_tr, x_ts, y_tr, y_ts, model_params)

Functions:
----------
- train_and_evaluate_models(x_tr, x_ts, y_tr, y_ts, model_params): Trains and evaluates models.
- train_deep_learning_models(x_tr, x_ts, y_tr, y_ts, model_params): Trains and evaluates deep learning models.
- rank_models(trained_models, x_ts, y_ts): Ranks the trained models based on their performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Flatten, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

def print_with_padding(message):
    """Prints a message with padding for better readability."""
    print(f"\n{'-'*10} {message} {'-'*10}\n")

def score_model(predictions, y_true):
    """Calculates the accuracy score of the model predictions."""
    return accuracy_score(y_true, predictions)

def train_and_evaluate_models(x_tr, x_ts, y_tr, y_ts, model_params):
    """
    Trains and evaluates multiple machine learning models.

    Parameters:
    x_tr (pd.DataFrame): Training features.
    x_ts (pd.DataFrame): Testing features.
    y_tr (pd.Series): Training labels.
    y_ts (pd.Series): Testing labels.
    model_params (dict): Dictionary containing model parameters.
    """
    label = ["Normal", "Anomalous"]
    models = {
        "Random Forest": RandomForestClassifier(**model_params["RandomForestClassifier"]),
        "K-Nearest Neighbors": KNeighborsClassifier(**model_params["KNeighborsClassifier"]),
        "Decision Tree": DecisionTreeClassifier(**model_params["DecisionTreeClassifier"]),
        "Logistic Regression": LogisticRegression(**model_params["LogisticRegression"]),
        "SVC": SVC(**model_params["SVC"]),
        "Naïve Bayes": GaussianNB(**model_params["GaussianNB"]),
    }
    trained_models = {}
    
    for model_name, model in models.items():
        print_with_padding(model_name)
        print("Computing....")
        model.fit(x_tr, y_tr)
        print("Done!")
        trained_models[model_name] = model

        predictions = model.predict(x_ts)
        print("MAE", mean_absolute_error(y_ts, predictions))
        print("Accuracy", accuracy_score(y_ts, predictions))
        print(
            "Precision",
            precision_score(
                y_ts, predictions, average="weighted", labels=np.unique(predictions)
            ),
        )
        print(
            "Recall",
            recall_score(
                y_ts, predictions, average="weighted", labels=np.unique(predictions)
            ),
        )
        print(
            "F1",
            f1_score(
                y_ts, predictions, average="weighted", labels=np.unique(predictions)
            ),
        )
        print(
            "ROC AUC",
            roc_auc_score(
                y_ts, predictions, average="weighted", labels=np.unique(predictions)
            ),
        )
        error = (predictions != y_ts).mean()
        print(f"Test error: {error:.1%}")

        cm = confusion_matrix(y_ts, predictions)
        cm = pd.DataFrame(cm, index=["0", "1"], columns=["0", "1"])
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            cm,
            cmap="Blues",
            linecolor="black",
            linewidth=1,
            annot=True,
            fmt="",
            xticklabels=label,
            yticklabels=label,
        )
        plt.title(model_name)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    # Rank models
    rank_models(trained_models, x_ts, y_ts)

def train_deep_learning_models(x_tr, x_ts, y_tr, y_ts, model_params):
    """
    Trains and evaluates deep learning models.

    Parameters:
    x_tr (pd.DataFrame): Training features.
    x_ts (pd.DataFrame): Testing features.
    y_tr (pd.Series): Training labels.
    y_ts (pd.Series): Testing labels.
    model_params (dict): Dictionary containing model parameters.
    """
    # Encode labels for binary classification
    label_encoder = LabelEncoder()
    y_tr_encoded = label_encoder.fit_transform(y_tr)
    y_ts_encoded = label_encoder.transform(y_ts)

    print_with_padding("Recurrent Neural Network (RNN)")
    RNN_model = Sequential()
    RNN_model.add(
        SimpleRNN(
            model_params["RNN"]["units"],
            input_shape=(x_tr.shape[1], 1),
            activation=model_params["RNN"]["activation"],
        )
    )
    RNN_model.add(Dense(units=1, activation="sigmoid"))
    RNN_model.compile(
        optimizer=Adam(learning_rate=model_params["RNN"]["learning_rate"]),
        loss=model_params["RNN"]["loss"],
        metrics=model_params["RNN"]["metrics"],
    )
    RNN_model.fit(
        x_tr,
        y_tr_encoded,
        epochs=model_params["RNN"]["epochs"],
        batch_size=model_params["RNN"]["batch_size"],
        validation_data=(x_ts, y_ts_encoded),
    )
    accuracy = RNN_model.evaluate(x_ts, y_ts_encoded)[1]
    print(f"Accuracy: {accuracy}")

    print_with_padding("Artificial Neural Network (ANN)")
    ANN_model = Sequential()
    for layer in model_params["ANN"]["layers"]:
        ANN_model.add(Dense(**layer["Dense"]))
    ANN_model.compile(
        optimizer=Adam(learning_rate=model_params["ANN"]["learning_rate"]),
        loss=model_params["ANN"]["loss"],
        metrics=model_params["ANN"]["metrics"],
    )
    ANN_model.fit(
        x_tr,
        y_tr_encoded,
        epochs=model_params["ANN"]["epochs"],
        batch_size=model_params["ANN"]["batch_size"],
        validation_data=(x_ts, y_ts_encoded),
    )
    accuracy = ANN_model.evaluate(x_ts, y_ts_encoded)[1]
    print(f"Accuracy: {accuracy}")

    print_with_padding("Convolutional Neural Network (CNN)")
    x_tr_np = x_tr.to_numpy().reshape(x_tr.shape[0], -1)
    x_ts_np = x_ts.to_numpy().reshape(x_ts.shape[0], -1)

    CNN_model = Sequential()
    CNN_model.add(Flatten(input_shape=(x_tr_np.shape[1],)))
    CNN_model.add(Dense(64, activation="relu"))
    CNN_model.add(Dense(32, activation="relu"))
    CNN_model.add(Dense(units=1, activation="sigmoid"))
    CNN_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    CNN_model.fit(
        x_tr_np,
        y_tr_encoded,
        epochs=30,
        batch_size=32,
        validation_data=(x_ts_np, y_ts_encoded),
    )
    accuracy = CNN_model.evaluate(x_ts_np, y_ts_encoded)[1]
    print(f"Accuracy: {accuracy}")

    print_with_padding("Long Short-Term Memory (LSTM)")
    LSTM_model = Sequential()
    LSTM_model.add(
        LSTM(
            model_params["LSTM"]["units"],
            input_shape=(x_tr.shape[1], 1),
            activation=model_params["LSTM"]["activation"],
        )
    )
    LSTM_model.add(Dense(units=1, activation="sigmoid"))
    LSTM_model.compile(
        optimizer=Adam(learning_rate=model_params["LSTM"]["learning_rate"]),
        loss=model_params["LSTM"]["loss"],
        metrics=model_params["LSTM"]["metrics"],
    )
    LSTM_model.fit(
        x_tr,
        y_tr_encoded,
        epochs=model_params["LSTM"]["epochs"],
        batch_size=model_params["LSTM"]["batch_size"],
        validation_data=(x_ts, y_ts_encoded),
    )
    accuracy = LSTM_model.evaluate(x_ts, y_ts_encoded)[1]
    print(f"Accuracy: {accuracy}")

def rank_models(trained_models, x_ts, y_ts):
    """
    Ranks the trained models based on their performance.

    Parameters:
    trained_models (dict): Dictionary of trained models.
    x_ts (pd.DataFrame): Testing features.
    y_ts (pd.Series): Testing labels.
    """
    print_with_padding("RANKING THE TRAINED MODELS ON THE MAE VALUE")

    model_preds = {name: model.predict(x_ts) for name, model in trained_models.items()}
    model_names = list(trained_models.keys())

    acc_score = []
    for model_name in model_names:
        acc = score_model(model_preds[model_name], y_ts)
        acc_score.append((model_name, acc))

    acc_scores_sorted = sorted(acc_score, key=lambda x: x[1], reverse=True)
    target_range = y_ts.max() - y_ts.min()

    for i, (model_name, acc) in enumerate(acc_scores_sorted):
        error_percent = (acc / target_range) * 100  # Calculate error percentage
        print(
            f"Rank {i + 1}: {model_name} - ACC: {acc:.4f} - Accuracy: {error_percent:.2f}%"
        )
