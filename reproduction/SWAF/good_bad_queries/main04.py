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


# (base) onyxia@vscode-tensorflow-gpu-923614-0:~/work/ML-SOC/reproduction/SWAF/good_bad_queries$ python main04.py 
# 2024-07-30 13:22:04.864689: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
# 2024-07-30 13:22:04.905784: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
# 2024-07-30 13:22:04.914964: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
# 2024-07-30 13:22:04.933566: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# 2024-07-30 13:22:06.396223: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
# /opt/conda/lib/python3.12/site-packages/keras/src/layers/core/embedding.py:90: UserWarning: Argument `input_length` is deprecated. Just remove it.
#   warnings.warn(
# 2024-07-30 13:22:10.236161: I tensorflow/core/common_runtime/gpu/gpu_device.cc:2021] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13775 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:af:00.0, compute capability: 7.5
# Epoch 1/10
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1722345732.552918    5295 service.cc:146] XLA service 0x7f3c7801d640 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
# I0000 00:00:1722345732.553019    5295 service.cc:154]   StreamExecutor device (0): Tesla T4, Compute Capability 7.5
# 2024-07-30 13:22:12.614170: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:268] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
# 2024-07-30 13:22:12.821521: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:531] Loaded cuDNN version 8900
# 2024-07-30 13:22:14.340045: E external/local_xla/xla/service/slow_operation_alarm.cc:65] Trying algorithm eng4{k11=2} for conv (f32[64,128,1,200]{3,2,1,0}, u8[0]{0}) custom-call(f32[64,128,1,196]{3,2,1,0}, f32[128,128,1,5]{3,2,1,0}), window={size=1x5}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBackwardInput", backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"cudnn_conv_backend_config":{"conv_result_scale":1,"activation_mode":"kNone","side_input_scale":0,"leakyrelu_alpha":0},"force_earliest_schedule":false} is taking a while...
# 2024-07-30 13:22:15.203145: E external/local_xla/xla/service/slow_operation_alarm.cc:133] The operation took 1.863206484s
# Trying algorithm eng4{k11=2} for conv (f32[64,128,1,200]{3,2,1,0}, u8[0]{0}) custom-call(f32[64,128,1,196]{3,2,1,0}, f32[128,128,1,5]{3,2,1,0}), window={size=1x5}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBackwardInput", backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"cudnn_conv_backend_config":{"conv_result_scale":1,"activation_mode":"kNone","side_input_scale":0,"leakyrelu_alpha":0},"force_earliest_schedule":false} is taking a while...
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# E0000 00:00:1722345736.018872    5295 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
# 2024-07-30 13:22:16.208471: E external/local_xla/xla/service/slow_operation_alarm.cc:65] Trying algorithm eng4{k11=1} for conv (f32[64,128,1,200]{3,2,1,0}, u8[0]{0}) custom-call(f32[64,128,1,196]{3,2,1,0}, f32[128,128,1,5]{3,2,1,0}), window={size=1x5}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBackwardInput", backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"cudnn_conv_backend_config":{"conv_result_scale":1,"activation_mode":"kNone","side_input_scale":0,"leakyrelu_alpha":0},"force_earliest_schedule":false} is taking a while...
# E0000 00:00:1722345736.520686    5295 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
# 2024-07-30 13:22:16.584862: E external/local_xla/xla/service/slow_operation_alarm.cc:133] The operation took 1.376474118s
# Trying algorithm eng4{k11=1} for conv (f32[64,128,1,200]{3,2,1,0}, u8[0]{0}) custom-call(f32[64,128,1,196]{3,2,1,0}, f32[128,128,1,5]{3,2,1,0}), window={size=1x5}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBackwardInput", backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"cudnn_conv_backend_config":{"conv_result_scale":1,"activation_mode":"kNone","side_input_scale":0,"leakyrelu_alpha":0},"force_earliest_schedule":false} is taking a while...
# I0000 00:00:1722345737.696524    5295 device_compiler.h:188] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
# 2911/2913 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9856 - loss: 0.04282024-07-30 13:22:29.809182: E external/local_xla/xla/service/slow_operation_alarm.cc:65] Trying algorithm eng3{k11=2} for conv (f32[128,128,1,5]{3,2,1,0}, u8[0]{0}) custom-call(f32[128,43,1,200]{3,2,1,0}, f32[128,43,1,196]{3,2,1,0}), window={size=1x196}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convForward", backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"cudnn_conv_backend_config":{"conv_result_scale":1,"activation_mode":"kNone","side_input_scale":0,"leakyrelu_alpha":0},"force_earliest_schedule":false} is taking a while...
# 2024-07-30 13:22:30.034588: E external/local_xla/xla/service/slow_operation_alarm.cc:133] The operation took 1.225535278s
# Trying algorithm eng3{k11=2} for conv (f32[128,128,1,5]{3,2,1,0}, u8[0]{0}) custom-call(f32[128,43,1,200]{3,2,1,0}, f32[128,43,1,196]{3,2,1,0}), window={size=1x196}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convForward", backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"cudnn_conv_backend_config":{"conv_result_scale":1,"activation_mode":"kNone","side_input_scale":0,"leakyrelu_alpha":0},"force_earliest_schedule":false} is taking a while...
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 22s 5ms/step - accuracy: 0.9856 - loss: 0.0427 - val_accuracy: 0.9979 - val_loss: 0.0050
# Epoch 2/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 0.9995 - loss: 0.0020 - val_accuracy: 1.0000 - val_loss: 4.4083e-05
# Epoch 3/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 12s 4ms/step - accuracy: 0.9998 - loss: 5.4922e-04 - val_accuracy: 0.9997 - val_loss: 5.1405e-04
# Epoch 4/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 0.9997 - loss: 8.1020e-04 - val_accuracy: 1.0000 - val_loss: 3.6359e-05
# Epoch 5/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 12s 4ms/step - accuracy: 1.0000 - loss: 2.0889e-04 - val_accuracy: 1.0000 - val_loss: 1.5416e-05
# Epoch 6/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 12s 4ms/step - accuracy: 0.9999 - loss: 4.1994e-04 - val_accuracy: 1.0000 - val_loss: 1.1475e-06
# Epoch 7/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 12s 4ms/step - accuracy: 0.9999 - loss: 5.0803e-04 - val_accuracy: 1.0000 - val_loss: 2.5123e-06
# Epoch 8/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 12s 4ms/step - accuracy: 0.9999 - loss: 2.2846e-04 - val_accuracy: 1.0000 - val_loss: 1.7186e-04
# Epoch 9/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 12s 4ms/step - accuracy: 1.0000 - loss: 4.0404e-05 - val_accuracy: 1.0000 - val_loss: 5.4248e-08
# Epoch 10/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 12s 4ms/step - accuracy: 0.9999 - loss: 5.6283e-04 - val_accuracy: 1.0000 - val_loss: 2.0515e-07
# 840/840 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step    
# Epoch 1/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 16s 5ms/step - accuracy: 0.9843 - loss: 0.0448 - val_accuracy: 0.9995 - val_loss: 0.0011
# Epoch 2/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 12s 4ms/step - accuracy: 0.9995 - loss: 0.0017 - val_accuracy: 0.9993 - val_loss: 0.0062
# Epoch 3/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 0.9995 - loss: 0.0017 - val_accuracy: 1.0000 - val_loss: 5.0317e-05
# Epoch 4/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 12s 4ms/step - accuracy: 0.9999 - loss: 5.2543e-04 - val_accuracy: 1.0000 - val_loss: 4.3995e-06
# Epoch 5/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 12s 4ms/step - accuracy: 0.9999 - loss: 1.8046e-04 - val_accuracy: 1.0000 - val_loss: 2.5126e-05
# Epoch 6/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 2.9075e-04 - val_accuracy: 1.0000 - val_loss: 2.5323e-07
# Epoch 7/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 5.6859e-07 - val_accuracy: 1.0000 - val_loss: 1.4748e-07
# Epoch 8/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 12s 4ms/step - accuracy: 1.0000 - loss: 1.3585e-07 - val_accuracy: 1.0000 - val_loss: 3.6797e-08
# Epoch 9/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 2.6380e-08 - val_accuracy: 1.0000 - val_loss: 4.2318e-08
# Epoch 10/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 12s 4ms/step - accuracy: 1.0000 - loss: 8.1153e-09 - val_accuracy: 1.0000 - val_loss: 3.8239e-09
# 840/840 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step    
# Epoch 1/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 16s 5ms/step - accuracy: 0.9837 - loss: 0.0456 - val_accuracy: 1.0000 - val_loss: 4.6555e-04
# Epoch 2/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 12s 4ms/step - accuracy: 0.9994 - loss: 0.0018 - val_accuracy: 1.0000 - val_loss: 2.4969e-04
# Epoch 3/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 12s 4ms/step - accuracy: 0.9999 - loss: 3.7820e-04 - val_accuracy: 1.0000 - val_loss: 7.7172e-06
# Epoch 4/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 12s 4ms/step - accuracy: 1.0000 - loss: 2.7365e-05 - val_accuracy: 1.0000 - val_loss: 8.3237e-07
# Epoch 5/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 7.1436e-07 - val_accuracy: 1.0000 - val_loss: 2.0510e-07
# Epoch 6/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 1.6352e-07 - val_accuracy: 1.0000 - val_loss: 5.7548e-08
# Epoch 7/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 3.6799e-08 - val_accuracy: 1.0000 - val_loss: 1.1027e-08
# Epoch 8/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 8.2269e-09 - val_accuracy: 1.0000 - val_loss: 2.4082e-09
# Epoch 9/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 1.9741e-09 - val_accuracy: 1.0000 - val_loss: 1.8029e-09
# Epoch 10/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 7.2028e-10 - val_accuracy: 1.0000 - val_loss: 3.7698e-10
# 840/840 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step    
# Epoch 1/10
# 2902/2913 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9853 - loss: 0.04222024-07-30 13:28:54.889410: E external/local_xla/xla/service/slow_operation_alarm.cc:65] Trying algorithm eng3{k11=2} for conv (f32[128,128,1,5]{3,2,1,0}, u8[0]{0}) custom-call(f32[128,45,1,200]{3,2,1,0}, f32[128,45,1,196]{3,2,1,0}), window={size=1x196}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convForward", backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"cudnn_conv_backend_config":{"conv_result_scale":1,"activation_mode":"kNone","side_input_scale":0,"leakyrelu_alpha":0},"force_earliest_schedule":false} is taking a while...
# 2024-07-30 13:28:55.191108: E external/local_xla/xla/service/slow_operation_alarm.cc:133] The operation took 1.301848458s
# Trying algorithm eng3{k11=2} for conv (f32[128,128,1,5]{3,2,1,0}, u8[0]{0}) custom-call(f32[128,45,1,200]{3,2,1,0}, f32[128,45,1,196]{3,2,1,0}), window={size=1x196}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convForward", backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"cudnn_conv_backend_config":{"conv_result_scale":1,"activation_mode":"kNone","side_input_scale":0,"leakyrelu_alpha":0},"force_earliest_schedule":false} is taking a while...
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 18s 5ms/step - accuracy: 0.9853 - loss: 0.0421 - val_accuracy: 0.9997 - val_loss: 4.6505e-04
# Epoch 2/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 0.9996 - loss: 0.0012 - val_accuracy: 1.0000 - val_loss: 1.5664e-05
# Epoch 3/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 7.1843e-05 - val_accuracy: 0.9990 - val_loss: 0.0019
# Epoch 4/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 0.9998 - loss: 6.3140e-04 - val_accuracy: 1.0000 - val_loss: 5.8609e-06
# Epoch 5/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 3.7821e-06 - val_accuracy: 1.0000 - val_loss: 5.6359e-07
# Epoch 6/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 7.7698e-07 - val_accuracy: 1.0000 - val_loss: 2.1651e-07
# Epoch 7/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 1.4216e-07 - val_accuracy: 1.0000 - val_loss: 3.3038e-08
# Epoch 8/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 3.6080e-08 - val_accuracy: 1.0000 - val_loss: 3.2368e-08
# Epoch 9/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 1.2559e-08 - val_accuracy: 1.0000 - val_loss: 3.9967e-09
# Epoch 10/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 2.4411e-09 - val_accuracy: 1.0000 - val_loss: 8.1433e-10
# 840/840 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step    
# Epoch 1/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 16s 5ms/step - accuracy: 0.9830 - loss: 0.0417 - val_accuracy: 0.9998 - val_loss: 4.6966e-04
# Epoch 2/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 0.9995 - loss: 0.0018 - val_accuracy: 1.0000 - val_loss: 1.5801e-05
# Epoch 3/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 0.9990 - loss: 0.0046 - val_accuracy: 0.9995 - val_loss: 9.2459e-04
# Epoch 4/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 12s 4ms/step - accuracy: 0.9998 - loss: 5.1193e-04 - val_accuracy: 1.0000 - val_loss: 2.2222e-05
# Epoch 5/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 7.5459e-05 - val_accuracy: 1.0000 - val_loss: 1.1954e-04
# Epoch 6/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 0.9999 - loss: 5.5354e-04 - val_accuracy: 1.0000 - val_loss: 2.5264e-05
# Epoch 7/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 0.9999 - loss: 1.6849e-04 - val_accuracy: 1.0000 - val_loss: 3.7889e-06
# Epoch 8/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 1.4362e-04 - val_accuracy: 1.0000 - val_loss: 3.3452e-06
# Epoch 9/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 1.8340e-06 - val_accuracy: 1.0000 - val_loss: 2.3959e-06
# Epoch 10/10
# 2913/2913 ━━━━━━━━━━━━━━━━━━━━ 11s 4ms/step - accuracy: 1.0000 - loss: 3.6687e-07 - val_accuracy: 1.0000 - val_loss: 8.6726e-08
# 840/840 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step    
# INFO:root:Mean accuracy: 0.9997393369674683, Standard Deviation: 8.492877907293207e-05
# INFO:root:Mean precision: 0.997711181640625, Standard Deviation: 0.0012125112116336823
# INFO:root:Mean recall: 0.9950125813484192, Standard Deviation: 0.0027420767582952976
# INFO:root:Mean F1-score: 0.9933483004570007, Standard Deviation: 0.0020719412714242935