"""
Script principal pour lancer les entraînements des modèles du stage.

Ce script lit la configuration depuis un fichier YAML, charge les données CSV,
 effectue la construction des caractéristiques,
et entraîne puis évalue plusieurs modèles de machine learning et de deep learning.
Les résultats des évaluations sont affichés
sous forme de métriques de performance et de matrices de confusion.

Fonctions:
----------
- print_with_padding(message): Affiche un message avec des tirets pour améliorer la lisibilité.
- load_config(): Charge la configuration à partir du fichier config.yaml.
- load_csv_data(filepath): Charge les données à partir d'un fichier CSV spécifié.
- build_features(data): Construit les caractéristiques à partir des données brutes.
- train_and_evaluate_models(x_tr, x_ts, y_tr, y_ts, model_params): Entraîne et évalue plusieurs
modèles de machine learning et de deep learning.

Étapes principales:
-------------------
1. Lecture de la configuration depuis config.yaml.
2. Chargement des données à partir du fichier CSV spécifié dans la configuration.
3. Construction des caractéristiques à partir des données brutes.
4. Division des données en ensembles d'entraînement et de test.
5. Entraînement et évaluation des modèles.

Les résultats des évaluations incluent les métriques de performance telles que MAE,
accuracy, precision, recall, F1 score, ROC AUC, et une matrice de confusion pour chaque modèle.
"""


import seaborn as sns
from src.config.load_config import load_config
from src.data.load_data import load_csv_data
from src.features.build_features import build_features
from src.models.train_and_evaluate import train_and_evaluate_models
from sklearn.model_selection import train_test_split
from src.features.build_features import filtrage_colonnes


def print_with_padding(message):
    print(f"\n{'-'*10} {message} {'-'*10}\n")


# ----------------CONFIG ------------------
print_with_padding("READ config.yaml")

# Usage example:
config = load_config()

# Access model hyperparameters from the configuration
# Access model hyperparameters from the configuration
model_params = {model: config["models"][model] for model in config["models"]}
CSIC_FILEPATH = config["data_path"]


# TODO mettre sur Minio
print_with_padding("CHARGEMENT DES DONNEES")
print_with_padding("READ CSV")


csic_data = load_csv_data(CSIC_FILEPATH)
print("Done!")

print("Number of samples:", csic_data.shape[0])
print("Number of features:", csic_data.shape[1])

# Data Visualization
print_with_padding("Data Visualization")

sns.set_style("darkgrid")
sns.countplot(data=csic_data, x="Unnamed: 0")

# Build features
X = filtrage_colonnes(csic_data)

X, y = build_features(csic_data)

# Split the dataset into training and testing sets
print("computing...")
x_tr, x_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=0)
print("Done!")
#
## Train and evaluate models
train_and_evaluate_models(x_tr, x_ts, y_tr, y_ts, model_params)
