"""
Module de tests pour les fonctions de construction de fonctionnalités.

Ce module contient des tests pour les fonctions `filtrage_colonnes` et `build_features` 
du module `src.features.build_features`.

Fixtures:
    - sample_data: Fournit des données d'exemple pour les tests.

Tests:
    - test_filtrage_colonnes: Vérifie que `filtrage_colonnes` filtre correctement les colonnes.
    - test_build_features: Vérifie que `build_features` construit correctement les fonctionnalités.
    - test_build_features_varied_data: Vérifie que `build_features` fonctionne correctement avec
    des données variées.

Usage:
    Pour exécuter les tests, utilisez la commande `pytest` dans le répertoire du projet.

Exemple:
    $ pytest tests/test_build_features.py
"""

import pytest
import pandas as pd
from src.features.build_features import build_features, filtrage_colonnes


# Fixture pour les données de test
@pytest.fixture
def sample_data():
    """
    Fixture that provides sample data for testing.

    Returns:
    pd.DataFrame: A dataframe containing sample data for testing.
    """
    data = {
        "Unnamed: 0": [0, 1],
        "Method": ["GET", "POST"],
        "User-Agent": ["Mozilla/5.0", "Mozilla/5.0"],
        "Pragma": ["no-cache", "no-cache"],
        "Cache-Control": ["no-cache", "no-cache"],
        "Accept": ["text/html", "application/xhtml+xml"],
        "Accept-encoding": ["gzip, deflate", "gzip, deflate"],
        "Accept-charset": ["ISO-8859-1", "ISO-8859-1"],
        "language": ["en-US,en;q=0.5", "en-US,en;q=0.5"],
        "host": ["example.com", "example.com"],
        "cookie": ["sessionid=12345", "sessionid=12345"],
        "content-type": ["text/html", "application/x-www-form-urlencoded"],
        "connection": ["keep-alive", "keep-alive"],
        "lenght": ["100", "200"],
        "content": ["<html></html>", "<html><body></body></html>"],
        "classification": ["benign", "malicious"],
        "URL": ["http://example.com", "http://example.com/test"]
    }
    return pd.DataFrame(data)


# Tests pour la fonction filtrage_colonnes
def test_filtrage_colonnes(sample_data1):
    """
    Tests the filtrage_colonnes function to ensure it correctly filters columns.

    Parameters:
    sample_data1 (pd.DataFrame): The sample data for testing.

    Asserts:
    - The resulting dataframe should not be empty.
    - The 'Method' column should be present in the features.
    - The 'URL' column should be present in the features.
    """
    X = filtrage_colonnes(sample_data1)
    assert not X.empty, "The features dataframe should not be empty."
    assert "Method" in X.columns, "The 'Method' column should be present in the features."
    assert "URL" in X.columns, "The 'URL' column should be present in the features."


# Tests pour la fonction build_features
def test_build_features(sample_data2):
    """
    Tests the build_features function to ensure it correctly builds features.

    Parameters:
    sample_data (pd.DataFrame): The sample data for testing.

    Asserts:
    - The resulting features dataframe should not be empty.
    - The target variable should have the same length as the input data.
    - Specific columns should be present in the features.
    - The target variable should be one-dimensional.
    - The target variable should contain only encoded values (0 and 1).
    """
    X, y = build_features(sample_data2)
    assert not X.empty, "The features dataframe should not be empty."
    assert len(y) == len(sample_data2), "The target variable should have the same length as \
        the input data."
    assert "content_length" in X.columns, "The 'content_length' column should be present in \
        the features."
    assert "count_dot_url" in X.columns, "The 'count_dot_url' column should be present in the \
        features."
    assert "count_dot_content" in X.columns, "The 'count_dot_content' column should be present \
        in the features."
    assert y.ndim == 1, "The target variable should be one-dimensional."
    assert set(y).issubset({0, 1}), "The target variable should contain only encoded values (0 \
        and 1)."


# Tests supplémentaires pour couvrir plus de cas
def test_build_features_varied_data():
    """
    Tests the build_features function with varied data to ensure robustness.

    Asserts:
    - The resulting features dataframe should not be empty.
    - The target variable should have the same length as the input data.
    - Specific columns should be present in the features.
    - The target variable should be one-dimensional.
    - The target variable should contain only encoded values (0 and 1).
    """
    varied_data = {
        "Unnamed: 0": [0, 1, 0],
        "Method": ["GET", "POST", "DELETE"],
        "User-Agent": ["Mozilla/5.0", "Mozilla/5.0", "Mozilla/4.0"],
        "Pragma": ["no-cache", "no-cache", ""],
        "Cache-Control": ["no-cache", "no-cache", "private"],
        "Accept": ["text/html", "application/xhtml+xml", "application/json"],
        "Accept-encoding": ["gzip, deflate", "gzip, deflate", ""],
        "Accept-charset": ["ISO-8859-1", "ISO-8859-1", "UTF-8"],
        "language": ["en-US,en;q=0.5", "en-US,en;q=0.5", "fr-FR,fr;q=0.5"],
        "host": ["example.com", "example.com", "test.com"],
        "cookie": ["sessionid=12345", "sessionid=12345", ""],
        "content-type": ["text/html", "application/x-www-form-urlencoded", "application/json"],
        "connection": ["keep-alive", "keep-alive", "close"],
        "lenght": ["100", "200", "300"],
        "content": ["<html></html>", "<html><body></body></html>",
                    "<html><head></head><body></body></html>"],
        "classification": ["benign", "malicious", "benign"],
        "URL": ["http://example.com", "http://example.com/test", "https://test.com/path"]
    }
    data_df = pd.DataFrame(varied_data)
    X, y = build_features(data_df)
    assert not X.empty, "The features dataframe should not be empty."
    assert len(y) == len(varied_data["Unnamed: 0"]), "The target variable should have the same \
        length as the input data."
    assert "content_length" in X.columns, "The 'content_length' column should be present in the \
        features."
    assert "count_dot_url" in X.columns, "The 'count_dot_url' column should be present in the \
        features."
    assert "count_dot_content" in X.columns, "The 'count_dot_content' column should be present in \
        the features."
    assert y.ndim == 1, "The target variable should be one-dimensional."
    assert set(y).issubset({0, 1}), "The target variable should contain only encoded values (0 \
        and 1)."


if __name__ == "__main__":
    pytest.main()
