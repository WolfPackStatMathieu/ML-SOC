import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from scikeras.wrappers import KerasClassifier
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)

def load_partial_data(badqueries_file, goodqueries_file, fraction=0.1):
    """
    Charge une fraction des données des fichiers fournis et les mélange.
    
    Args:
    badqueries_file (str): Chemin vers le fichier des mauvaises requêtes.
    goodqueries_file (str): Chemin vers le fichier des bonnes requêtes.
    fraction (float): Fraction des données à charger.
    
    Returns:
    np.ndarray, np.ndarray: URLs et étiquettes correspondantes.
    """
    # Lire les fichiers ligne par ligne
    with open(badqueries_file, 'r') as file:
        bad_queries = file.readlines()
    with open(goodqueries_file, 'r') as file:
        good_queries = file.readlines()
    
    # Prendre un échantillon proportionnel
    bad_queries_sample = np.random.choice(bad_queries, size=int(fraction * len(bad_queries)), replace=False)
    good_queries_sample = np.random.choice(good_queries, size=int(fraction * len(good_queries)), replace=False)
    
    # Créer des DataFrames
    bad_queries_df = pd.DataFrame(bad_queries_sample, columns=["URL"])
    good_queries_df = pd.DataFrame(good_queries_sample, columns=["URL"])
    
    # Ajouter les étiquettes
    bad_queries_df['label'] = 1
    good_queries_df['label'] = 0
    
    # Combiner et mélanger les données
    data = pd.concat([bad_queries_df, good_queries_df])
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Mélanger les données
    
    return data['URL'].values, data['label'].values

class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Transformateur personnalisé pour prétraiter les données de requêtes HTTP.
    Convertit les requêtes en séquences ASCII.
    """
    def __init__(self, max_length=200):
        self.max_length = max_length
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        ascii_data = np.zeros((len(X), self.max_length), dtype=int)
        for i, url in enumerate(X):
            ascii_values = [ord(char) for char in url[:self.max_length]]
            ascii_data[i, :len(ascii_values)] = ascii_values
        return ascii_data

def create_cnn_model(input_length=200):
    """
    Crée un modèle de réseau de neurones convolutionnel (CNN) pour la classification.
    
    Args:
    input_length (int): Longueur des séquences d'entrée.
    
    Returns:
    model: Modèle CNN compilé.
    """
    model = Sequential([
        Embedding(input_dim=128, output_dim=128, input_length=input_length),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Créer le classificateur Keras avec la fonction create_cnn_model
cnn_classifier = KerasClassifier(model=create_cnn_model, input_length=200, epochs=10, batch_size=64, verbose=1)

# Charger les données
X, y = load_partial_data('badqueries.txt', 'goodqueries.txt', fraction=0.1)

# Créer le pipeline
pipeline = Pipeline([
    ('preprocessor', Preprocessor(max_length=200)),
    ('classifier', cnn_classifier)
])

# Initialiser StratifiedKFold
kf = StratifiedKFold(n_splits=5)

# Validation croisée avec plusieurs métriques
scoring = ['accuracy', 'precision', 'recall', 'f1']
results = cross_validate(pipeline, X, y, cv=kf, scoring=scoring)

# Afficher les résultats
logging.info(f'Mean accuracy: {results["test_accuracy"].mean()}, Standard Deviation: {results["test_accuracy"].std()}')
logging.info(f'Mean precision: {results["test_precision"].mean()}, Standard Deviation: {results["test_precision"].std()}')
logging.info(f'Mean recall: {results["test_recall"].mean()}, Standard Deviation: {results["test_recall"].std()}')
logging.info(f'Mean F1-score: {results["test_f1"].mean()}, Standard Deviation: {results["test_f1"].std()}')
