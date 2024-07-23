import time
import os
import numpy as np
import s3fs
import pandas as pd
import tempfile
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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


def check_aws_credentials():
    print("check_aws_credentials")
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_session_token = os.getenv('AWS_SESSION_TOKEN')
    aws_region = os.getenv('AWS_DEFAULT_REGION')
    aws_s3_endpoint = os.getenv('AWS_S3_ENDPOINT')
    if not aws_access_key_id:
        raise EnvironmentError("AWS_ACCESS_KEY_ID is not set.")
    if not aws_secret_access_key:
        raise EnvironmentError("AWS_SECRET_ACCESS_KEY is not set.")
    if not aws_session_token:
        raise EnvironmentError("AWS_SESSION_TOKEN is not set.")
    if not aws_region:
        raise EnvironmentError("AWS_DEFAULT_REGION is not set.")
    if not aws_s3_endpoint:
        raise EnvironmentError("AWS_S3_ENDPOINT is not set.")


def train_random_forest(x_train, y_train, n_estimators, max_leaf_nodes):
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

    pipe_gscv.fit(x_train, y_train)
    log_gsvc_to_mlflow(gscv=pipe_gscv, mlflow_experiment_name="random_forest")

    best_model = pipe_gscv.best_estimator_

    print("Done!")
    return best_model


def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)

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

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal (class 0)", "Anomalous (class 1)"],
        )
    )

    return predictions


def upload_to_s3(file_path):
    
    check_aws_credentials()

    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_session_token = os.getenv('AWS_SESSION_TOKEN')
    aws_region = os.getenv('AWS_DEFAULT_REGION')
    aws_s3_endpoint = os.getenv('AWS_S3_ENDPOINT')

    print("AWS Access Key ID:", aws_access_key_id)
    print("AWS Secret Access Key:", aws_secret_access_key)
    print("AWS Session Token:", aws_session_token)
    print("AWS Default Region:", aws_region)
    print("AWS S3 Endpoint:", aws_s3_endpoint)

    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': f'https://{aws_s3_endpoint}'},
        key=aws_access_key_id,
        secret=aws_secret_access_key,
        token=aws_session_token
    )

    try:
        with open(file_path, 'rb') as f:
            fs.put(file_path, f's3://mthomassin/output/{os.path.basename(file_path)}')
        print(f"Successfully uploaded {file_path} to s3://mthomassin/output/{os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")


def plot_confusion_matrix(
    y_test, predictions, labels=None, output_path="output/fig/confusion_matrix_rf.png"
):
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()

    upload_to_s3(output_path)


def save_pipeline_to_s3(pipeline):
    print("Start save pipeline to s3")
    check_aws_credentials()
    
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_session_token = os.getenv('AWS_SESSION_TOKEN')
    aws_s3_endpoint = os.getenv('AWS_S3_ENDPOINT')

    print(f"AWS_ACCESS_KEY_ID: {aws_access_key_id}")
    print(f"AWS_SECRET_ACCESS_KEY: {aws_secret_access_key}")
    print(f"AWS_SESSION_TOKEN: {aws_session_token}")
    print(f"AWS_S3_ENDPOINT: {aws_s3_endpoint}")

    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': f'https://{aws_s3_endpoint}'},
        key=aws_access_key_id,
        secret=aws_secret_access_key,
        token=aws_session_token
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        pipeline_path = os.path.join(tmpdirname, 'complete_preprocessor_pipeline.pkl')
        print(f"pipeline_path : {pipeline_path}")
        joblib.dump(pipeline, pipeline_path)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        s3_path = f's3://mthomassin/preprocessor/complete_preprocessor_pipeline_{timestamp}.pkl'
        print(f"Uploading {pipeline_path} to {s3_path}")
        fs.put(pipeline_path, s3_path)
        print("Upload complete")


def model_random_forest(data, params):
    check_aws_credentials()
    print("Building features and preprocessing...")
    preprocessor, numeric_transformer, categorical_transformer = preprocessing_pipeline()

    feature_builder = preprocessor.named_steps['feature_builder']
    X_transformed, y = feature_builder.fit_transform(data)
    print(f"Features after feature_builder.transform: {X_transformed.shape}")
    print(f"Numeric features: {feature_builder.numeric_features}")
    print(f"Categorical features: {feature_builder.categorical_features}")

    print("Colonnes après feature_builder.transform:")
    print(X_transformed.columns)

    print(f"Target variable 'y' après feature_builder.transform: {y}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, feature_builder.numeric_features),
            ('cat', categorical_transformer, feature_builder.categorical_features)
        ])

    X = preprocessor.fit_transform(X_transformed)
    print(f"Features after preprocessor.transform: {X.shape}")
    print("Colonnes après preprocessor.transform:")
    print(f"type(X): {type(X)}")

    complete_pipeline = Pipeline(steps=[
        ('feature_builder', feature_builder),
        ('preprocessor', preprocessor)
    ])
    
    save_pipeline_to_s3(complete_pipeline)

    print("Computing train and test split...")
    x_tr, x_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=0)
    print(f"Training set shape: {x_tr.shape}, Testing set shape: {x_ts.shape}")
    print("Done!")

    start_time = time.time()
    model = train_random_forest(x_tr, y_tr, params['n_estimators'], params['max_leaf_nodes'])
    end_time = time.time()
    print(f"RANDOM FOREST Execution time: {end_time - start_time:.2f} seconds")
    predictions = evaluate_model(model, x_ts, y_ts)
    plot_confusion_matrix(y_ts, predictions, output_path=f"output/fig/confusion_matrix_rf_{params['n_estimators']}_{params['max_leaf_nodes']}.png")
