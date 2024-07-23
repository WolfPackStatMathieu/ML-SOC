"""
Ce module fournit des utilitaires pour enregistrer des modèles scikit-learn 
entraînés avec GridSearchCV dans MLflow.

Fonctions:
    - log_gsvc_to_mlflow: Enregistre un objet GridSearchCV entraîné comme une expérience MLflow.
"""

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from src.config.load_config import load_config

def log_gsvc_to_mlflow(gscv, mlflow_experiment_name):
    """
    Enregistre un objet GridSearchCV entraîné avec scikit-learn en tant qu'expérience MLflow.

    Args:
        gscv (GridSearchCV): L'objet GridSearchCV entraîné.
        mlflow_experiment_name (str): Le nom de l'expérience MLflow.

    Cette fonction configure le contexte MLflow, enregistre les hyperparamètres
    et les métriques de performance
    pour chaque combinaison d'hyperparamètres testée, et sauvegarde le modèle
    entraîné en tant qu'artefact MLflow.
    """
    try:
        # Configuration du contexte MLflow avec le nom de l'expérience
        mlflow.set_experiment(experiment_name=mlflow_experiment_name)

        # Boucle sur chaque combinaison d'hyperparamètres testée par GridSearchCV
        for run_idx in range(len(gscv.cv_results_["params"])):
            # Nom de l'exécution MLflow pour cette combinaison d'hyperparamètres
            run_name = f"run {run_idx}"
            with mlflow.start_run(run_name=run_name):
                # Enregistrement des hyperparamètres
                params = gscv.cv_results_["params"][run_idx]
                for param in params:
                    mlflow.log_param(param, params[param])

                # Enregistrement des métriques de performance
                scores = [
                    score
                    for score in gscv.cv_results_
                    if "mean_test" in score or "std_test" in score
                ]
                for score in scores:
                    mlflow.log_metric(score, gscv.cv_results_[score][run_idx])

                # Enregistrement du modèle en tant qu'artefact
                mlflow.sklearn.log_model(gscv, "gscv_model")

                # Enregistrement de l'URL des données d'entraînement
                config = load_config()
                CSIC_FILEPATH = config["data_path"]
                mlflow.log_param("data_url", CSIC_FILEPATH)

    except mlflow.exceptions.MlflowException as e:
        # Gestion des exceptions spécifiques à MLflow
        print(f"MLflowException: {e}")
    except Exception as e:
        # Gestion des autres exceptions
        print(f"Exception: {e}")
