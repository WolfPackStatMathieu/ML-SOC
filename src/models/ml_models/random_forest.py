import time
import os
import numpy as np
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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
from src.mlflow.mlflow_utils import log_gsvc_to_mlflow
from src.features.preprocessing import preprocessing_pipeline
from sklearn.compose import ColumnTransformer
import joblib


def train_random_forest(x_train, y_train, n_estimators, max_leaf_nodes):
    """
    Train a Random Forest classifier on the provided training data.

    Parameters:
    x_train (DataFrame): Features for training.
    y_train (Series): Target variable for training.
    n_estimators (int): Number of trees in the forest.
    max_leaf_nodes (int): Maximum number of leaf nodes in each tree.

    Returns:
    RandomForestClassifier: Trained Random Forest model.
    """
    random_forest_model = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes)
    pipe_rf = Pipeline(
        [("classifier", random_forest_model)]
    )

    param_grid = {
        "classifier__n_estimators": [n_estimators],
        "classifier__max_leaf_nodes": [max_leaf_nodes],
    }

    pipe_gscv = GridSearchCV(
        pipe_rf,
        param_grid=param_grid,
        scoring=["accuracy", "precision", "recall", "f1"],
        refit="f1",
        cv=5,
        n_jobs=5,
        verbose=1,
    )
    print("Computing....")

    # Fit the model using GridSearchCV
    pipe_gscv.fit(x_train, y_train)
    # Log results to MLflow
    log_gsvc_to_mlflow(gscv=pipe_gscv, mlflow_experiment_name="random_forest")

    # Evaluate the best model
    best_model = pipe_gscv.best_estimator_

    print("Done!")
    return best_model


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
    error_rt = (predictions != y_test).mean()
    print(f"Test error: {error_rt:.1%}")

    # Afficher le rapport de classification
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal (class 0)", "Anomalous (class 1)"],
        )
    )

    return predictions


def plot_confusion_matrix(
    y_test, predictions, labels=None, output_path="output/fig/confusion_matrix_rf.png"
):
    """
    Plot the confusion matrix for the model predictions.

    Parameters:
    y_test (Series): True labels for the test data.
    predictions (ndarray): Predicted labels by the model.
    labels (list): List of label names for the confusion matrix.
    """
    # Correctly set labels if not provided
    if labels is None:
        labels = ["Normal", "Anomalous"]

    cm = confusion_matrix(y_test, predictions)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(10, 10))
    sns.heatmap(
        cm_df,
        cmap="Blues",
        linecolor="black",
        linewidth=1,
        annot=True,
        fmt="",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(output_path)
    plt.show()


def save_pipeline_to_s3(pipeline):
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_session_token = os.getenv('AWS_SESSION_TOKEN')
    aws_region = os.getenv('AWS_DEFAULT_REGION')

    with tempfile.TemporaryDirectory() as tmpdirname:
        pipeline_path = os.path.join(tmpdirname, 'complete_preprocessor_pipeline.pkl')
        joblib.dump(pipeline, pipeline_path)
        # Utilisation de mc pour copier le fichier
        os.system(f"AWS_ACCESS_KEY_ID={aws_access_key_id} AWS_SECRET_ACCESS_KEY={aws_secret_access_key} AWS_SESSION_TOKEN={aws_session_token} AWS_DEFAULT_REGION={aws_region} mc cp {pipeline_path} s3/mthomassin/preprocessor/complete_preprocessor_pipeline.pkl")




def model_random_forest(data, params):
    """
    Train, evaluate, and plot the Random Forest model.

    This function trains a Random Forest classifier on the dataset, evaluates it on the test
    data, and plots the confusion matrix.

    Parameters:
    data (pd.DataFrame): The raw data.
    params (dict): Hyperparameters for Random Forest.
    """
    print("Building features and preprocessing...")
    preprocessor, numeric_transformer, categorical_transformer = preprocessing_pipeline()

    # Apply feature builder separately
    feature_builder = preprocessor.named_steps['feature_builder']
    X_transformed, y = feature_builder.fit_transform(data)
    print(f"Features after feature_builder.transform: {X_transformed.shape}")
    print(f"Numeric features: {feature_builder.numeric_features}")
    print(f"Categorical features: {feature_builder.categorical_features}")

    print("Colonnes après feature_builder.transform:")
    print(X_transformed.columns)

    print(f"Target variable 'y' après feature_builder.transform: {y}")

    # Create ColumnTransformer with the correct features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, feature_builder.numeric_features),
            ('cat', categorical_transformer, feature_builder.categorical_features)
        ])

    # Fit and transform the data with ColumnTransformer
    X = preprocessor.fit_transform(X_transformed)
    print(f"Features after preprocessor.transform: {X.shape}")
    print("Colonnes après preprocessor.transform:")
    print(X.columns if hasattr(X, 'columns') else 'Not a DataFrame')

    # Sauvegarder le pipeline complet de prétraitement (incluant FeatureBuilder et ColumnTransformer)
    complete_pipeline = Pipeline(steps=[
        ('feature_builder', feature_builder),
        ('preprocessor', preprocessor)
    ])
    
    
    # Appel de la fonction dans le script principal
    save_pipeline_to_s3(complete_pipeline)

    # Split the dataset into training and testing sets
    print("Computing train and test split...")
    x_tr, x_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=0)
    print(f"Training set shape: {x_tr.shape}, Testing set shape: {x_ts.shape}")
    print("Done!")

    start_time = time.time()
    model = train_random_forest(x_tr, y_tr, params['n_estimators'], params['max_leaf_nodes'])
    end_time = time.time()
    print(f"RANDOM FOREST Execution time: {end_time - start_time:.2f} seconds")
    predictions = evaluate_model(model, x_ts, y_ts)
    plot_confusion_matrix(y_ts, predictions)
