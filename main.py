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

import os
# import pandas as pd
# import numpy as np
# import seaborn as sns
from src.config.load_config import load_config
from src.data.load_data import load_csv_data
# from src.features.build_features import build_features
# from src.features.preprocessing import build_features

from sklearn.model_selection import train_test_split
from src.models.ml_models.random_forest import model_random_forest
from src.models.ml_models.knn import model_knn
from src.models.ml_models.decision_tree import model_decision_tree
from src.models.ml_models.logistic_regression import model_logistic_regression
from src.models.ml_models.svm import model_svm
from src.models.ml_models.naive_bayes import model_naive_bayes

from src.models.dl_models.rnn import model_rnn
from src.models.dl_models.ann import model_ann
from src.models.dl_models.cnn import model_cnn
from src.models.dl_models.lstm import model_lstm
from src.utils.print_utils import print_with_padding


print_with_padding("MLflow Tracking Server")
# Automatic discovery : if MLFlow has been launched before Jupyter/VSCode
if "MLFLOW_TRACKING_URI" in os.environ:
    print(os.environ["MLFLOW_TRACKING_URI"])
else:
    print("MLflow was not automatically discovered, a tracking URI must be provided manually.")


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

# Supprimer la colonne 'Unnamed: 0'
csic_data = csic_data.drop(columns=['Unnamed: 0'], errors='ignore')

print("Colonnes après suppression de 'Unnamed: 0':")
print(csic_data.columns)
# print("Number of samples:", csic_data.shape[0])
# print("Number of features:", csic_data.shape[1])
# Data Visualization
# print_with_padding("Data Visualization")
# sns.set_style("darkgrid")
# sns.countplot(data=csic_data, x="Unnamed: 0")


# # Build features
# X, y = build_features(csic_data)

# Split the dataset into training and testing sets
# print("computing train and test split...")
# x_tr, x_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=0)
# print("Done!")
# #
# ## Train and evaluate models
# train_and_evaluate_models(x_tr, x_ts, y_tr, y_ts, model_params)
# Reset index for testing set
# # Display the unique values in the training target variable and its name
# y_tr = pd.Series(y_tr)

# print(y_tr.unique())
# print(y_tr.name)

# x_ts = x_ts.reset_index(drop=True)
# y_ts = pd.Series(y_ts).reset_index(drop=True)


# Display mean of features for each class in the testing set
# for k in range(np.unique(y_ts).size):
#     print('mean of class ' + str(k) + ':\n', x_ts[y_ts == k].mean(axis=0))


print_with_padding("MACHINE LEARNING MODELS")

# Timing and running models
print_with_padding("RANDOM FOREST")
print("Colonnes avant l'appel de 'model_random_forest':")
print(csic_data.columns)

model_random_forest(csic_data, model_params["RandomForestClassifier"])

print_with_padding("K-NEAREST NEIGHBOR")
# model_knn(x_tr, y_tr, x_ts, y_ts, model_params["KNeighborsClassifier"])
 
print_with_padding("DECISION TREE")
# model_decision_tree(x_tr, y_tr, x_ts, y_ts, model_params["DecisionTreeClassifier"])

print_with_padding("LOGISTIC REGRESSION")
# model_logistic_regression(x_tr, y_tr, x_ts, y_ts, model_params["LogisticRegression"])

print_with_padding("SUPPORT VECTOR MACHINE (SVM)")
# model_svm(x_tr, y_tr, x_ts, y_ts, model_params["SVC"])

print_with_padding("NAÏVE BAYES")
# model_naive_bayes(x_tr, y_tr, x_ts, y_ts, model_params["GaussianNB"])

print_with_padding("DEEP LEARNING")

print_with_padding("RECURRENT NEURAL NETWORK")
# model_rnn(x_tr, y_tr, x_ts, y_ts, model_params["RNN"])

print_with_padding("ARTIFICIAL NEURAL NETWORK")
# model_ann(x_tr, y_tr, x_ts, y_ts, model_params["ANN"])

print_with_padding("CONVOLUTIONAL NEURAL NETWORK")
# model_cnn(x_tr, y_tr, x_ts, y_ts, model_params["CNN"])

print_with_padding("LONG SHORT TERM MEMORY")
# model_lstm(x_tr, y_tr, x_ts, y_ts, model_params["LSTM"])
