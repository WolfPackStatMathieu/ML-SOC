"""
Module for making predictions using a pre-trained model and pre-processing pipeline.
"""

import os
import joblib
import mlflow
from src.config.load_config import load_config
from src.data.load_data import load_csv_data
from src.utils.print_utils import print_with_padding

print_with_padding("MLflow Tracking Server")
# Automatic discovery: if MLFlow has been lancé before Jupyter/VSCode
if "MLFLOW_TRACKING_URI" in os.environ:
    print(os.environ["MLFLOW_TRACKING_URI"])
else:
    print("MLflow was not automatically discovered, a tracking URI must be provided manually.")

MODEL_NAME = "random_forest_detection"
VERSION = 2

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{VERSION}"
)

print_with_padding("READ config.yaml")
config = load_config()
print(config)
# Access model hyperparameters from the configuration
model_params = {model: config["models"][model] for model in config["models"]}
CSIC_FILEPATH = config["data_path"]
print(model_params["RandomForestClassifier"])
print_with_padding("CHARGEMENT DES DONNEES")
print_with_padding("READ CSV")

csic_data = load_csv_data(CSIC_FILEPATH)

# Vérification des colonnes et types de données
print("Columns in the dataset: ", csic_data.columns)
print("Data types in the dataset: ", csic_data.dtypes)

# Étape 4: Extraction des lignes pour la prédiction
rows_for_prediction = csic_data.head(5)

print(rows_for_prediction)
# Étape 5: Chargement du pipeline de prétraitement sauvegardé
complete_pipeline = joblib.load('complete_preprocessor_pipeline.pkl')

# Étape 6: Prétraitement des données pour la prédiction
print("Building features and preprocessing...")

# Transformation des données avec le pipeline complet
try:
    # Appliquer FeatureBuilder pour générer les fonctionnalités manquantes
    feature_builder = complete_pipeline.named_steps['feature_builder']
    X_transformed, _ = feature_builder.transform(rows_for_prediction)
    print(f"Features after feature_builder.transform: {X_transformed.shape}")

    # Appliquer le reste du pipeline de prétraitement
    preprocessor = complete_pipeline.named_steps['preprocessor']
    X = preprocessor.transform(X_transformed)
    print(f"Features after complete_pipeline.transform: {X.shape}")

    # Étape 7: Prédiction
    predictions = model.predict(X)

    # Affichage des prédictions
    print_with_padding("PREDICTIONS")
    print(predictions)
    urls = rows_for_prediction['URL'].tolist()
    for i, prediction in enumerate(predictions):
        url = urls[i].split(" ")[0]  # Extraire l'URL avant " HTTP/1.1"
        if prediction == 1:
            print("Requête anormale")
            print(url)
        else:
            print("Requête normale")
            print(url)

except ValueError as e:
    print(f"ValueError during transformation or prediction: {e}")
except KeyError as e:
    print(f"KeyError during transformation or prediction: {e}")
except Exception as e:
    print(f"Unexpected error during transformation or prediction: {e}")
