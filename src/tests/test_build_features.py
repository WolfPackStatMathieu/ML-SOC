import pytest
import pandas as pd
from src.features.build_features import build_features, filtrage_colonnes

@pytest.fixture
def sample_data():
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

def test_filtrage_colonnes(sample_data):
    X = filtrage_colonnes(sample_data)
    assert not X.empty, "The features dataframe should not be empty."
    assert "Method" in X.columns, "The 'Method' column should be present in the features."
    assert "URL" in X.columns, "The 'URL' column should be present in the features."

def test_build_features(sample_data):
    X, y = build_features(sample_data)
    assert not X.empty, "The features dataframe should not be empty."
    assert len(y) == len(sample_data), "The target variable should have the same length as the input data."
    assert "content_length" in X.columns, "The 'content_length' column should be present in the features."
    assert "count_dot_url" in X.columns, "The 'count_dot_url' column should be present in the features."
    assert "count_dot_content" in X.columns, "The 'count_dot_content' column should be present in the features."
    assert y.ndim == 1, "The target variable should be one-dimensional."
    assert set(y).issubset({0, 1}), "The target variable should contain only encoded values (0 and 1)."

if __name__ == "__main__":
    pytest.main()
