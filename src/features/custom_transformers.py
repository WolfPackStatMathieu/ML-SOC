"""
Ce module définit une classe de transformateur personnalisé pour construire des caractéristiques
à partir de données brutes en utilisant la fonction `build_features` définie dans `src.features.build_features`.

Classes:
    - FeatureBuilder: Un transformateur personnalisé qui utilise `build_features` pour transformer les données.

Importe:
    - BaseEstimator, TransformerMixin: Classes de base pour les estimators et les transformateurs de scikit-learn.
    - build_features: Fonction pour prétraiter et extraire des caractéristiques des données brutes.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from src.features.build_features import build_features

class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Un transformateur personnalisé pour construire des caractéristiques à partir de données brutes.

    Cette classe utilise la fonction `build_features` pour transformer les données d'entrée en 
    caractéristiques prêtes à être utilisées pour l'entraînement de modèles.

    Attributes:
        numeric_features (list): Liste des caractéristiques numériques extraites.
        categorical_features (list): Liste des caractéristiques catégorielles extraites.
    """
    def __init__(self):
        """
        Initialise le FeatureBuilder avec des listes vides pour les caractéristiques numériques et catégorielles.
        """
        self.numeric_features = []
        self.categorical_features = []

    def fit(self, X, y=None):
        """
        Cette méthode est requise par scikit-learn, mais n'effectue aucune opération pour ce transformateur.

        Parameters:
        X (pd.DataFrame): Les données d'entraînement.
        y (pd.Series, optional): La variable cible. Par défaut à None.

        Returns:
        self: L'objet lui-même.
        """
        return self

    def transform(self, X):
        """
        Transforme les données d'entrée en utilisant la fonction `build_features`.

        Parameters:
        X (pd.DataFrame): Les données brutes à transformer.

        Returns:
        pd.DataFrame: Les caractéristiques transformées.
        pd.Series: La variable cible.
        """
        X_transformed, y, self.numeric_features, self.categorical_features = build_features(X)
        
        # Affiche les caractéristiques extraites
        print(f"Numeric features: {self.numeric_features}")
        print(f"Categorical features: {self.categorical_features}")
        print(f"Transformed features shape: {X_transformed.shape}")
        
        return X_transformed, y

    def get_feature_names_out(self, input_features=None):
        """
        Retourne les noms des caractéristiques extraites.

        Parameters:
        input_features (list, optional): Les noms des caractéristiques d'entrée. Par défaut à None.

        Returns:
        list: Les noms des caractéristiques extraites.
        """
        return self.numeric_features + self.categorical_features
