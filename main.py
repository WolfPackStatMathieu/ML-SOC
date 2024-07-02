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
4. Division des données en ensembles d'entraînement et detest.
5. Entraînement et évaluation des modèles.

Les résultats des évaluations incluent les métriques de performance telles que MAE,
accuracy, precision, recall, F1 score, ROC AUC, et une matrice de confusion pour chaque modèle.
"""

import pandas as pd
import numpy as np
import seaborn as sns
from src.config.load_config import load_config
from src.data.load_data import load_csv_data
from src.features.build_features import build_features

from sklearn.model_selection import train_test_split
from src.features.build_features import filtrage_colonnes
from src.models.ml_models.random_forest import model_random_forest
from src.models.ml_models.knn import model_knn
from src.models.ml_models.decision_tree import model_decision_tree
from src.models.ml_models.logistic_regression import model_logistic_regression
from src.models.ml_models.svm import model_svm
from src.models.ml_models.naive_bayes import model_naive_bayes

from src.models.dl_models.rnn import model_rnn

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
print("computing train and test split...")
x_tr, x_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=0)
print("Done!")
# #
# ## Train and evaluate models
# train_and_evaluate_models(x_tr, x_ts, y_tr, y_ts, model_params)
# Reset index for testing set
# Display the unique values in the training target variable and its name
y_tr = pd.Series(y_tr)

print(y_tr.unique())
print(y_tr.name)

x_ts = x_ts.reset_index(drop=True)
y_ts = pd.Series(y_ts).reset_index(drop=True)
# Display mean of features for each class in the testing set
for k in range(np.unique(y_ts).size):
    print('mean of class ' + str(k) + ':\n', x_ts[y_ts == k].mean(axis=0))


print_with_padding("MACHINE LEARNING MODELS")

print_with_padding("RANDOM FOREST")
# model_random_forest(x_tr, y_tr, x_ts, y_ts)

print_with_padding("K-NEAREST NEIGHBOR")
#model_knn(x_tr, y_tr, x_ts, y_ts)

print_with_padding("DECISION TREE")
#model_decision_tree(x_tr, y_tr, x_ts, y_ts)


print_with_padding("Logistic Regression")
# model_params = {
#     'random_state': 42,
#     'max_iter': 1000
# }
# model_logistic_regression(x_tr, y_tr, x_ts, y_ts, model_params)

print_with_padding("Support Vector Machine (SVM)")
# model_params = {
#     'C': 1.0,
#     'kernel': 'rbf',
#     'gamma': 'scale'
# }
# model_svm(x_tr, y_tr, x_ts, y_ts, model_params)

print_with_padding("Naïve Bayes")
# model_naive_bayes(x_tr, y_tr, x_ts, y_ts)


print_with_padding("DEEP LEARNING")

print_with_padding("Recurrent Neural Network")
model_rnn(x_tr, y_tr, x_ts, y_ts, model_params["RNN"])

