import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
import logging
import csv

# Configuration du logging
logging.basicConfig(level=logging.INFO)

def load_partial_data(badqueries_file, goodqueries_file, fraction=0.1):
    """
    Charge un extrait proportionnel des fichiers de requêtes mauvaises et bonnes.
    
    Args:
    badqueries_file (str): Chemin vers le fichier contenant les mauvaises requêtes.
    goodqueries_file (str): Chemin vers le fichier contenant les bonnes requêtes.
    fraction (float): Fraction des données à charger.

    Returns:
    tuple: Tableaux numpy des URL et des labels.
    """
    
    def read_lines(file_path):
        """
        Lit un fichier ligne par ligne.
        
        Args:
        file_path (str): Chemin vers le fichier à lire.
        
        Returns:
        list: Liste des lignes du fichier.
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return lines
    
    # Lecture des fichiers ligne par ligne
    bad_queries = read_lines(badqueries_file)
    good_queries = read_lines(goodqueries_file)
    
    # Prendre un échantillon proportionnel
    bad_queries_sample = np.random.choice(bad_queries, size=int(len(bad_queries) * fraction), replace=False)
    good_queries_sample = np.random.choice(good_queries, size=int(len(good_queries) * fraction), replace=False)
    
    # Ajouter les étiquettes et créer des DataFrames
    bad_queries_sample = pd.DataFrame(bad_queries_sample, columns=["URL"])
    bad_queries_sample['label'] = 1
    good_queries_sample = pd.DataFrame(good_queries_sample, columns=["URL"])
    good_queries_sample['label'] = 0
    
    # Combiner et mélanger les données
    data = pd.concat([bad_queries_sample, good_queries_sample])
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Mélanger les données
    
    return data['URL'].values, data['label'].values

def preprocess_data(urls):
    """
    Convertit les URL en séquences ASCII avec une longueur maximale de 200 caractères.
    
    Args:
    urls (array): Tableau d'URL à convertir.
    
    Returns:
    array: Tableau numpy des séquences ASCII.
    """
    max_length = 200  # Troncature ou padding des séquences à une longueur maximale de 200
    ascii_data = np.zeros((len(urls), max_length), dtype=int)
    
    for i, url in enumerate(urls):
        ascii_values = [ord(char) if ord(char) < 128 else 127 for char in url[:max_length]]
        ascii_data[i, :len(ascii_values)] = ascii_values
    
    return ascii_data

def create_cnn_model(input_length):
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

def balance_classes(X, y):
    """
    Équilibre les classes dans les données d'entraînement par sur-échantillonnage.
    
    Args:
    X (array): Données d'entrée.
    y (array): Étiquettes correspondantes.
    
    Returns:
    tuple: Données et étiquettes équilibrées.
    """
    # Séparer les classes
    X_neg = X[y == 0]
    y_neg = y[y == 0]
    X_pos = X[y == 1]
    y_pos = y[y == 1]
    
    # Équilibrer les classes par sur-échantillonnage
    X_pos, y_pos = resample(X_pos, y_pos, replace=True, n_samples=len(X_neg), random_state=42)
    
    # Combiner les données équilibrées
    X_balanced = np.vstack((X_neg, X_pos))
    y_balanced = np.concatenate((y_neg, y_pos))
    
    return X_balanced, y_balanced

# Charger les données avec une fraction de 10%
X, y = load_partial_data('badqueries.txt', 'goodqueries.txt', fraction=0.1)

# Initialiser StratifiedKFold pour la validation croisée
kf = StratifiedKFold(n_splits=5)

# Listes pour stocker les métriques de performance
accuracies = []
precisions = []
recalls = []
f1_scores = []
errors = []

# K-Fold Cross Validation
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Prétraiter les données en convertissant les URL en séquences ASCII
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    
    # Équilibrer les classes dans les données d'entraînement
    X_train_balanced, y_train_balanced = balance_classes(X_train, y_train)
    
    # Créer le modèle CNN
    model = create_cnn_model(input_length=X_train_balanced.shape[1])
    
    # Entraîner le modèle
    model.fit(X_train_balanced, y_train_balanced, epochs=10, batch_size=64, validation_split=0.1, verbose=1)
    
    # Évaluer le modèle sur les données de test
    scores = model.evaluate(X_test, y_test, verbose=0)
    accuracies.append(scores[1])
    
    # Prédictions sur les données de test
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    # Enregistrer les erreurs
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            errors.append((X[test_index[i]], y_test[i], y_pred[i]))
    
    # Calcul des métriques de performance
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    f1_score = tf.keras.metrics.AUC(curve='PR')
    
    precision.update_state(y_test, y_pred)
    recall.update_state(y_test, y_pred)
    f1_score.update_state(y_test, y_pred)
    
    precisions.append(precision.result().numpy())
    recalls.append(recall.result().numpy())
    f1_scores.append(f1_score.result().numpy())

# Calculer les moyennes et écarts-types des métriques
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_precision = np.mean(precisions)
std_precision = np.std(precisions)
mean_recall = np.mean(recalls)
std_recall = np.std(recalls)
mean_f1_score = np.mean(f1_scores)
std_f1_score = np.std(f1_scores)

# Afficher les résultats
logging.info(f'Mean accuracy: {mean_accuracy}, Standard Deviation: {std_accuracy}')
logging.info(f'Mean precision: {mean_precision}, Standard Deviation: {std_precision}')
logging.info(f'Mean recall: {mean_recall}, Standard Deviation: {std_recall}')
logging.info(f'Mean F1-score: {mean_f1_score}, Standard Deviation: {std_f1_score}')

# Enregistrer les erreurs dans un fichier CSV
with open('errors.csv', 'w', newline='') as csvfile:
    error_writer = csv.writer(csvfile)
    error_writer.writerow(['URL', 'True Label', 'Predicted Label'])
    for error in errors:
        url, true_label, pred_label = error
        error_writer.writerow([url, true_label, pred_label])
