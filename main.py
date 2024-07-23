"""
Script pour charger la configuration, les données, et exécuter un modèle de forêt aléatoire
 en utilisant MLflow pour le tracking des expériences.

Ce script utilise les étapes suivantes :
1. Chargement de la configuration à partir d'un fichier YAML.
2. Chargement des données à partir d'un fichier CSV.
3. Exécution d'un modèle de forêt aléatoire avec les paramètres spécifiés.
4. Utilisation de MLflow pour le suivi des expériences.

Dépendances :
    - argparse
    - os
    - mlflow
    - src.config.load_config
    - src.data.load_data
    - src.models.ml_models.random_forest
    - src.utils.print_utils

Instructions :
    Utilisez ce script en ligne de commande avec les arguments nécessaires :
    python script.py --n_estimators <nombre_d_estimateurs> --max_leaf_nodes <nombre_max_de_feuilles>
"""
import argparse
import os
from src.config.load_config import load_config
from src.data.load_data import load_csv_data
from src.models.ml_models.random_forest import model_random_forest
from src.utils.print_utils import print_with_padding


def main(n_estimators, max_leaf_nodes):
    """
    Fonction principale pour charger la configuration, les données,
     et exécuter le modèle de forêt aléatoire.

    Args:
        n_estimators (int): Nombre d'estimateurs pour le modèle de forêt aléatoire.
        max_leaf_nodes (int): Nombre maximum de nœuds feuilles pour le modèle de forêt aléatoire.
    """
    # Affichage du serveur de tracking MLflow
    print_with_padding("MLflow Tracking Server")
    if "MLFLOW_TRACKING_URI" in os.environ:
        print(os.environ["MLFLOW_TRACKING_URI"])
    else:
        print(
            "MLflow was not automatically discovered, a tracking URI must be provided manually."
        )

    # Lecture et affichage de la configuration
    print_with_padding("READ config.yaml")
    config = load_config()
    print(config)
    model_params = {model: config["models"][model] for model in config["models"]}
    csic_filepath = config["data_path"]
    print(f"CSIC_FILEPATH: {csic_filepath}")
    print(
        f'model_params["RandomForestClassifier"]: {model_params["RandomForestClassifier"]}'
    )

    # Chargement et affichage des données
    print_with_padding("CHARGEMENT DES DONNEES")
    print_with_padding("READ CSV")
    csic_data = load_csv_data(csic_filepath)
    csic_data = csic_data.drop(columns=["Unnamed: 0"], errors="ignore")
    print("Colonnes après suppression de 'Unnamed: 0':")
    print(csic_data.columns)
    print(f"csic_data.shape: {csic_data.shape}")

    # Exécution du modèle de forêt aléatoire
    print_with_padding("MACHINE LEARNING MODELS")
    print_with_padding("RANDOM FOREST")
    print("Colonnes avant l'appel de 'model_random_forest':")
    print(f"csic_data.columns: {csic_data.columns}")

    # Convertir les arguments en dictionnaire
    params = {"n_estimators": n_estimators, "max_leaf_nodes": max_leaf_nodes}

    model_random_forest(csic_data, params)


if __name__ == "__main__":
    import mlflow

    # Affichage du serveur de tracking MLflow
    print_with_padding("MLflow Tracking Server")
    if "MLFLOW_TRACKING_URI" in os.environ:
        print(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    else:
        print(
            "MLflow was not automatically discovered, a tracking URI must be provided manually."
        )

    # Définition et analyse des arguments de la ligne de commande
    parser = argparse.ArgumentParser(
        description="Script pour charger les données, configurer et\
         exécuter un modèle de forêt aléatoire."
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        required=True,
        help="Nombre d'estimateurs pour le modèle de forêt aléatoire.",
    )
    parser.add_argument(
        "--max_leaf_nodes",
        type=int,
        required=True,
        help="Nombre maximum de nœuds feuilles pour le modèle de forêt aléatoire.",
    )
    args = parser.parse_args()

    print(args)

    # Appel de la fonction principale avec les arguments fournis
    main(args.n_estimators, args.max_leaf_nodes)
