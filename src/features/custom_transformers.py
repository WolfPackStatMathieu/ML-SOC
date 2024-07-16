from sklearn.base import BaseEstimator, TransformerMixin
from src.features.build_features import build_features


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_features = []
        self.categorical_features = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed, y, self.numeric_features, self.categorical_features = build_features(X)
        print(f"Numeric features: {self.numeric_features}")
        print(f"Categorical features: {self.categorical_features}")
        print(f"Transformed features shape: {X_transformed.shape}")
        return X_transformed, y

    def get_feature_names_out(self, input_features=None):
        return self.numeric_features + self.categorical_features