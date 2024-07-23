"""
Ce module contient des fonctions pour entraîner un modèle de forêt aléatoire, évaluer le modèle, 
et enregistrer les résultats sur AWS S3. Le script utilise MLflow pour le suivi des expériences 
et intègre diverses étapes de prétraitement des données.

Fonctions incluses :
- check_aws_credentials : Vérifie la présence des identifiants AWS dans les variables d'environnement.
- train_random_forest : Entraîne un modèle de forêt aléatoire avec validation croisée et journalise le modèle sur MLflow.
- evaluate_model : Évalue les performances du modèle sur un jeu de test.
- upload_to_s3 : Télécharge un fichier local vers un emplacement S3 spécifié.
- plot_confusion_matrix : Génère et télécharge une matrice de confusion pour les prédictions du modèle.
- save_pipeline_to_s3 : Enregistre un pipeline de prétraitement sur S3.
- model_random_forest : Exécute l'ensemble du processus de modélisation pour la forêt aléatoire, de la préparation des données à l'évaluation.

Dépendances :
    - argparse
    - os
    - numpy
    - s3fs
    - pandas
    - tempfile
    - subprocess
    - matplotlib.pyplot
    - seaborn
    - datetime
    - sklearn.pipeline
    - sklearn.model_selection
    - sklearn.ensemble
    - sklearn.metrics
    - src.mlflow.mlflow_utils
    - src.features.preprocessing
    - sklearn.compose
    - joblib
"""

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
    """
    Vérifie que les identifiants AWS nécessaires sont définis dans les variables d'environnement.

    Cette fonction vérifie la présence des variables d'environnement suivantes :
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_SESSION_TOKEN
    - AWS_DEFAULT_REGION
    - AWS_S3_ENDPOINT

    Si l'une de ces variables n'est pas définie, une erreur d'environnement est levée.

    Raises:
        EnvironmentError: Si une des variables d'environnement nécessaires n'est pas définie.
    """
    print("check_aws_credentials")
    
    # Récupération des variables d'environnement pour les identifiants AWS
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_session_token = os.getenv('AWS_SESSION_TOKEN')
    aws_region = os.getenv('AWS_DEFAULT_REGION')
    aws_s3_endpoint = os.getenv('AWS_S3_ENDPOINT')
    
    # Vérification de chaque variable d'environnement
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
    """
    Entraîne un modèle de forêt aléatoire avec une validation croisée en utilisant GridSearchCV et enregistre
    les résultats dans MLflow.

    Args:
        x_train (array-like): Les caractéristiques d'entraînement.
        y_train (array-like): Les étiquettes de classe d'entraînement.
        n_estimators (int): Le nombre d'arbres dans la forêt aléatoire.
        max_leaf_nodes (int): Le nombre maximum de nœuds feuilles dans chaque arbre.

    Returns:
        Pipeline: Le meilleur modèle trouvé par GridSearchCV.
    """
    # Initialiser le modèle de forêt aléatoire avec les hyperparamètres spécifiés
    random_forest_model = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes)
    pipe_rf = Pipeline(
        [("classifier", random_forest_model)]
    )

    # Définir la grille de paramètres pour la validation croisée
    param_grid = {
        "classifier__n_estimators": [n_estimators],
        "classifier__max_leaf_nodes": [max_leaf_nodes],
    }

    # Configurer GridSearchCV avec la pipeline et la grille de paramètres
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

    # Entraîner le modèle en utilisant GridSearchCV
    pipe_gscv.fit(x_train, y_train)

    # Enregistrer les résultats de GridSearchCV dans MLflow
    log_gsvc_to_mlflow(gscv=pipe_gscv, mlflow_experiment_name="random_forest")

    # Récupérer le meilleur modèle trouvé par GridSearchCV
    best_model = pipe_gscv.best_estimator_

    print("Done!")

    # Retourner le meilleur modèle
    return best_model


def evaluate_model(model, x_test, y_test):
    """
    Évalue les performances d'un modèle de classification sur un jeu de test.

    Args:
        model (Pipeline): Le modèle de classification entraîné.
        x_test (array-like): Les caractéristiques de test.
        y_test (array-like): Les étiquettes de classe de test.

    Returns:
        array: Les prédictions du modèle sur le jeu de test.
    """
    # Prédire les étiquettes pour les données de test
    predictions = model.predict(x_test)

    # Calculer et afficher les métriques d'évaluation
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

    # Calculer et afficher le taux d'erreur de test
    error_rt = (predictions != y_test).mean()
    print(f"Test error: {error_rt:.1%}")

    # Afficher un rapport de classification détaillé
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal (class 0)", "Anomalous (class 1)"],
        )
    )

    # Retourner les prédictions
    return predictions


def upload_to_s3(local_path, s3_path):
    """
    Télécharge un fichier local vers un emplacement S3 spécifié.

    Args:
        local_path (str): Chemin local du fichier à télécharger.
        s3_path (str): Chemin S3 où télécharger le fichier.
    """
    # Vérifie que les identifiants AWS sont définis dans les variables d'environnement
    check_aws_credentials()

    # Récupère les identifiants AWS et l'endpoint S3 à partir des variables d'environnement
    aws_access_key_id = os.getenv('aws_access_key_id_project')  # Clé d'accès publique
    aws_secret_access_key = os.getenv('aws_secret_access_key_project')  # Clé d'accès secrète
    aws_s3_endpoint = os.getenv('AWS_S3_ENDPOINT')

    # Affiche les identifiants AWS et l'endpoint S3
    print("AWS Access Key ID:", aws_access_key_id)
    print("AWS Secret Access Key:", aws_secret_access_key)
    print("AWS S3 Endpoint:", aws_s3_endpoint)

    # Initialise le système de fichiers S3
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': f'https://{aws_s3_endpoint}'},
        key=aws_access_key_id,
        secret=aws_secret_access_key
    )

    try:
        # Génère un timestamp et crée le chemin complet du fichier sur S3
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        s3_full_path = f's3://{s3_path.replace(".", f"_{timestamp}.")}'

        # Ouvre le fichier local en mode binaire et le télécharge sur S3
        with open(local_path, 'rb') as f:
            fs.put(local_path, s3_full_path)

        # Affiche un message de succès
        print(f"Successfully uploaded {local_path} to {s3_full_path}")
    except Exception as e:
        # Gère les exceptions et affiche un message d'erreur
        print(f"Error uploading to S3: {str(e)}")



def plot_confusion_matrix(
    y_test, predictions, labels=None, output_path="output/fig/confusion_matrix_rf.png"
):
    """
    Génère une matrice de confusion à partir des prédictions du modèle et des étiquettes de test,
    et télécharge l'image générée vers un emplacement S3 spécifié.

    Args:
        y_test (array-like): Les étiquettes de classe réelles du jeu de test.
        predictions (array-like): Les prédictions du modèle sur le jeu de test.
        labels (list, optional): Les étiquettes des classes. Par défaut, ["Normal", "Anomalous"].
        output_path (str, optional): Le chemin de sortie pour l'image de la matrice de confusion. Par défaut, "output/fig/confusion_matrix_rf.png".
    """
    if labels is None:
        labels = ["Normal", "Anomalous"]

    # Calculer la matrice de confusion
    cm = confusion_matrix(y_test, predictions)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Configurer la figure pour la matrice de confusion
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

    # Créer un répertoire temporaire pour sauvegarder l'image avant de la télécharger
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_output_path = os.path.join(tmpdirname, "confusion_matrix_rf.png")
        
        # Sauvegarder l'image de la matrice de confusion
        plt.savefig(temp_output_path)
        
        # Afficher l'image
        plt.show()

        # Afficher le chemin de sortie
        print(f"Confusion matrix output_path: {output_path}")
        
        # Télécharger l'image vers S3
        upload_to_s3(temp_output_path, output_path)


def save_pipeline_to_s3(pipeline):
    """
    Sauvegarde un pipeline de prétraitement dans un fichier et le télécharge vers un emplacement S3 spécifié.

    Args:
        pipeline (Pipeline): Le pipeline de prétraitement à sauvegarder.
    """
    print("Start save pipeline to s3")
    
    # Vérifie que les identifiants AWS sont définis dans les variables d'environnement
    check_aws_credentials()

    # Récupère les identifiants AWS et l'endpoint S3 à partir des variables d'environnement
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_session_token = os.getenv('AWS_SESSION_TOKEN')
    aws_s3_endpoint = os.getenv('AWS_S3_ENDPOINT')

    # Affiche les identifiants AWS et l'endpoint S3
    print(f"AWS_ACCESS_KEY_ID: {aws_access_key_id}")
    print(f"AWS_SECRET_ACCESS_KEY: {aws_secret_access_key}")
    print(f"AWS_SESSION_TOKEN: {aws_session_token}")
    print(f"AWS_S3_ENDPOINT: {aws_s3_endpoint}")

    # Initialise le système de fichiers S3
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': f'https://{aws_s3_endpoint}'},
        key=aws_access_key_id,
        secret=aws_secret_access_key,
        token=aws_session_token
    )

    # Crée un répertoire temporaire pour sauvegarder le pipeline
    with tempfile.TemporaryDirectory() as tmpdirname:
        pipeline_path = os.path.join(tmpdirname, 'complete_preprocessor_pipeline.pkl')
        
        # Sauvegarde le pipeline dans un fichier
        print(f"pipeline_path : {pipeline_path}")
        joblib.dump(pipeline, pipeline_path)

        # Génère un timestamp et crée le chemin complet du fichier sur S3
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        s3_path = f's3://mthomassin/preprocessor/complete_preprocessor_pipeline_{timestamp}.pkl'
        
        # Affiche le chemin de destination sur S3
        print(f"Uploading {pipeline_path} to {s3_path}")
        
        # Télécharge le fichier sauvegardé vers S3
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
