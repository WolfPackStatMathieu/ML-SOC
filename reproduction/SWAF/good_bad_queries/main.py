import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

# Configurer la journalisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Réduire la verbosité des logs TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Afficher le nombre de GPU disponibles
logging.info("Checking GPU availability...")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Fonction pour charger les fichiers badqueries.txt et goodqueries.txt par morceaux
def load_data_in_chunks(badqueries_file, goodqueries_file, chunk_size=10000):
    logging.info(f"Loading data from {badqueries_file} and {goodqueries_file} in chunks of {chunk_size}...")
    
    def read_file_in_chunks(file, label, chunk_size):
        with open(file, 'r') as f:
            chunk = []
            for line in f:
                chunk.append(line.strip())
                if len(chunk) == chunk_size:
                    yield pd.DataFrame({'URL': chunk, 'label': label})
                    chunk = []
            if chunk:
                yield pd.DataFrame({'URL': chunk, 'label': label})

    bad_chunks = read_file_in_chunks(badqueries_file, 1, chunk_size)
    good_chunks = read_file_in_chunks(goodqueries_file, 0, chunk_size)
    
    for bad_chunk, good_chunk in zip(bad_chunks, good_chunks):
        yield pd.concat([bad_chunk, good_chunk], ignore_index=True)

# Fonction pour prétraiter les données
def preprocess_data(data):
    logging.info("Preprocessing data...")
    http_requests = data['URL'].str.strip().values  # Supprimer les espaces autour des URLs
    labels = data['label'].values
    logging.info("Preprocessing complete.")
    return http_requests, labels

# Fonction pour convertir les requêtes HTTP en séquences de codes ASCII
def ascii_embedding(http_requests):
    logging.info("Converting HTTP requests to ASCII sequences...")
    max_length = max(len(req) for req in http_requests)
    data = np.zeros((len(http_requests), max_length), dtype=int)
    for i, req in enumerate(http_requests):
        ascii_codes = [ord(char) for char in req]
        data[i, :len(ascii_codes)] = ascii_codes
    logging.info("Conversion to ASCII complete.")
    return data, max_length

# Définir les hyperparamètres pour le modèle CNN
embedding_dim = 64  # Dimension des vecteurs d'embedding
batch_size = 64  # Taille des lots pour l'entraînement
activation_function = 'relu'  # Fonction d'activation ReLU
loss_function = SparseCategoricalCrossentropy()  # Fonction de perte pour la classification
dropout_rate = 0.5  # Taux de dropout pour éviter le surapprentissage
num_classes = 2  # Nombre de classes pour la classification

# Fonction pour créer le modèle CNN
def create_cnn_model(input_length):
    logging.info("Creating CNN model...")
    model = Sequential([
        Embedding(input_dim=256, output_dim=embedding_dim, input_length=input_length),  # Couche d'embedding pour convertir les codes ASCII en vecteurs d'embedding
        Conv1D(filters=128, kernel_size=5, activation=activation_function),  # Couche de convolution 1D pour extraire les caractéristiques
        MaxPooling1D(pool_size=2),  # Couche de max pooling pour réduire la dimensionnalité
        Flatten(),  # Couche pour aplatir la sortie des couches précédentes en un vecteur 1D
        Dense(128, activation=activation_function),  # Couche entièrement connectée avec 128 neurones et activation ReLU
        Dropout(dropout_rate),  # Couche de dropout pour éviter le surapprentissage
        Dense(num_classes, activation='softmax')  # Couche de sortie avec activation softmax pour la classification
    ])
    logging.info("CNN model created.")
    return model

# Initialiser la validation croisée en 5 plis
kf = KFold(n_splits=5)
accuracies = []  # Liste pour stocker les précisions de chaque pli
precisions = []  # Liste pour stocker les précisions (precision) de chaque pli
recalls = []  # Liste pour stocker les rappels (recall) de chaque pli
f1_scores = []  # Liste pour stocker les scores F1 de chaque pli

# Charger et préparer les données par morceaux
logging.info("Starting K-Fold Cross Validation...")
chunk_size = 10000  # Taille des morceaux
for data_chunk in load_data_in_chunks('badqueries.txt', 'goodqueries.txt', chunk_size=chunk_size):
    http_requests, labels = preprocess_data(data_chunk)
    data, max_length = ascii_embedding(http_requests)
    
    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        logging.info(f"Processing fold {fold + 1}...")
        X_train, X_test = data[train_index], data[test_index]  # Diviser les données en indices de train et de test
        y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]  # Diviser les étiquettes en train et test
        
        model = create_cnn_model(max_length)  # Créer une nouvelle instance du modèle CNN pour chaque pli
        optimizer = Adam()  # Recréer l'optimiseur pour chaque pli
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])  # Compiler le modèle avec l'optimiseur et la fonction de perte
        
        logging.info("Training model...")
        try:
            model.fit(X_train, y_train, batch_size=batch_size, epochs=10, validation_split=0.2)  # Entraîner le modèle avec les données d'entraînement
        except Exception as e:
            logging.error(f"Error during training: {e}")
            break
        
        logging.info("Evaluating model...")
        try:
            y_pred = np.argmax(model.predict(X_test), axis=1)  # Prédire les étiquettes pour les données de test
            accuracy = accuracy_score(y_test, y_pred)  # Calculer la précision
            precision = precision_score(y_test, y_pred)  # Calculer la précision (precision)
            recall = recall_score(y_test, y_pred)  # Calculer le rappel (recall)
            f1 = f1_score(y_test, y_pred)  # Calculer le score F1

            accuracies.append(accuracy)  # Stocker la précision de chaque pli dans la liste accuracies
            precisions.append(precision)  # Stocker la précision (precision) de chaque pli dans la liste precisions
            recalls.append(recall)  # Stocker le rappel (recall) de chaque pli dans la liste recalls
            f1_scores.append(f1)  # Stocker le score F1 de chaque pli dans la liste f1_scores

            logging.info(f"Fold {fold + 1} results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            break

# Après la boucle, imprimer les métriques moyennes et leurs écarts-types
logging.info("Cross Validation complete. Calculating mean and standard deviation of metrics...")
try:
    print(f'Mean accuracy: {np.mean(accuracies)}, Standard Deviation: {np.std(accuracies)}')
    print(f'Mean precision: {np.mean(precisions)}, Standard Deviation: {np.std(precisions)}')
    print(f'Mean recall: {np.mean(recalls)}, Standard Deviation: {np.std(recalls)}')
    print(f'Mean F1-score: {np.mean(f1_scores)}, Standard Deviation: {np.std(f1_scores)}')
except Exception as e:
    logging.error(f"Error during final metrics calculation: {e}")
