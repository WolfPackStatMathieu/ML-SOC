
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
    Create a preprocessing pipeline for the dataset.

    Returns:
    sklearn.pipeline.Pipeline: A pipeline that preprocesses the dataset.
    """
    feature_builder = FeatureBuilder()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Ensure categorical features are treated as strings for imputation
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, feature_builder.numeric_features),
            ('cat', categorical_transformer, feature_builder.categorical_features)
        ])

    pipeline = Pipeline(steps=[
        ('feature_builder', feature_builder),
        ('preprocessor', preprocessor)
    ])

    return pipeline, numeric_transformer, categorical_transformer
