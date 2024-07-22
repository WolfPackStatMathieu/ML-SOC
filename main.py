import argparse
import os
from src.config.load_config import load_config
from src.data.load_data import load_csv_data
from src.models.ml_models.random_forest import model_random_forest
from src.utils.print_utils import print_with_padding


def main(n_estimators, max_leaf_nodes):
    print_with_padding("MLflow Tracking Server")
    if "MLFLOW_TRACKING_URI" in os.environ:
        print(os.environ["MLFLOW_TRACKING_URI"])
    else:
        print("MLflow was not automatically discovered, a tracking URI must be provided manually.")

    print_with_padding("READ config.yaml")
    config = load_config()
    print(config)
    model_params = {model: config["models"][model] for model in config["models"]}
    CSIC_FILEPATH = config["data_path"]
    print(model_params["RandomForestClassifier"])
    print_with_padding("CHARGEMENT DES DONNEES")
    print_with_padding("READ CSV")

    csic_data = load_csv_data(CSIC_FILEPATH)
    csic_data = csic_data.drop(columns=['Unnamed: 0'], errors='ignore')
    print("Colonnes après suppression de 'Unnamed: 0':")
    print(csic_data.columns)

    print_with_padding("MACHINE LEARNING MODELS")
    print_with_padding("RANDOM FOREST")
    print("Colonnes avant l'appel de 'model_random_forest':")
    print(csic_data.columns)

    # Convertir les arguments en dictionnaire
    params = {
        "n_estimators": n_estimators,
        "max_leaf_nodes": max_leaf_nodes
    }

    model_random_forest(csic_data, params)


if __name__ == "__main__":
    import mlflow
    print_with_padding("MLflow Tracking Server")
    if "MLFLOW_TRACKING_URI" in os.environ:
        print(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    else:
        print("MLflow was not automatically discovered, a tracking URI must be provided manually.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, required=True, help='Number of estimators')
    parser.add_argument('--max_leaf_nodes', type=int, required=True, help='Maximum leaf nodes')
    args = parser.parse_args()
    print(args)
    main(args.n_estimators, args.max_leaf_nodes)
