
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.features.custom_transformers import FeatureBuilder


def filtrage_colonnes(data):
    """
    Preprocess and filter columns from the raw data.

    Parameters:
    data (pd.DataFrame): The raw data.

    Returns:
    pd.DataFrame: Filtered and renamed columns.
    """
    feature_names = [
        "Method",
        "User-Agent",
        "Pragma",
        "Cache-Control",
        "Accept",
        "Accept-encoding",
        "Accept-charset",
        "language",
        "host",
        "cookie",
        "content-type",
        "connection",
        "lenght",
        "content",
        "URL",
    ]
    X = data[feature_names]
    X = X.rename(columns={"lenght": "content_length"})
    return X


def build_features(data):
    """
    Preprocess and extract features from the raw data.

    Parameters:
    data (pd.DataFrame): The raw data.

    Returns:
    pd.DataFrame: The features.
    pd.Series: The target variable.
    """
    X = filtrage_colonnes(data)
    selected_features = [
        "Class",
        "Method",
        "host",
        "cookie",
        "Accept",
        "content_length",
        "content",
        "URL",
    ]
    return X[selected_features], data["target_column"]


def preprocessing_pipeline():
    """
    Crée une pipeline de prétraitement pour le jeu de données.

    Returns:
    sklearn.pipeline.Pipeline: Une pipeline qui prétraite le jeu de données.
    tuple: La pipeline de transformation numérique et la pipeline de transformation catégorielle.
    """
    # Initialise le constructeur de caractéristiques personnalisé
    feature_builder = FeatureBuilder()
    
    # Crée une pipeline pour les transformations des caractéristiques numériques
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Remplace les valeurs manquantes par la moyenne
        ('scaler', StandardScaler())  # Normalise les caractéristiques numériques
    ])

    # Crée une pipeline pour les transformations des caractéristiques catégorielles
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Remplace les valeurs manquantes par 'missing'
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encode les caractéristiques catégorielles en utilisant l'encodage one-hot
    ])

    # Crée un préprocesseur qui applique les transformations numériques et catégorielles
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, feature_builder.numeric_features),  # Applique les transformations numériques
            ('cat', categorical_transformer, feature_builder.categorical_features)  # Applique les transformations catégorielles
        ])

    # Crée une pipeline complète qui inclut le constructeur de caractéristiques et le préprocesseur
    pipeline = Pipeline(steps=[
        ('feature_builder', feature_builder),  # Construit les caractéristiques à partir des données brutes
        ('preprocessor', preprocessor)  # Applique les transformations préprocesseur
    ])

    # Retourne la pipeline complète ainsi que les transformateurs numériques et catégoriels
    return pipeline, numeric_transformer, categorical_transformer

