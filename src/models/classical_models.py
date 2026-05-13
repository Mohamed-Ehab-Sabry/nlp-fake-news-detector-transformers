"""
Classical Machine Learning Models for Fake News Detection

Models:
- K-Nearest Neighbors (KNN)
- Multinomial Naive Bayes
- Random Forest
- XGBoost

Features:
- TF-IDF feature extraction
- Dimensionality reduction (SVD/LSA)
- Model persistence and loading
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from joblib import load, dump

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)


class ClassicalModels:
    """Container for classical ML models with TF-IDF features"""

    def __init__(
        self,
        max_features: int = 10000,
        dimensionality_reduction: bool = False,
        n_components: int = 1000,
    ):
        """
        Initialize classical models pipeline

        Args:
            max_features: Maximum TF-IDF features
            dimensionality_reduction: Whether to apply SVD
            n_components: Number of SVD components
        """
        self.max_features = max_features
        self.dimensionality_reduction = dimensionality_reduction
        self.n_components = n_components

        # Feature extraction
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            lowercase=True,
            stop_words="english",
        )

        # Dimensionality reduction
        self.svd = None
        if dimensionality_reduction:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)

        # Models
        self.knn = None
        self.naive_bayes = None
        self.random_forest = None
        self.xgboost = None

    def extract_features(self, texts: list) -> np.ndarray:
        """
        Extract TF-IDF features from texts

        Args:
            texts: List of text strings

        Returns:
            Feature matrix (sparse or dense)
        """
        X = self.tfidf.fit_transform(texts)

        if self.dimensionality_reduction and self.svd is not None:
            X = self.svd.fit_transform(X)
            print(f"Dimensionality reduced: {X.shape}")

        return X

    def train_knn(
        self, X_train: np.ndarray, y_train: np.ndarray, n_neighbors: int = 5
    ) -> "KNeighborsClassifier":
        """Train KNN classifier"""
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        self.knn.fit(X_train, y_train)
        print(f"KNN trained with k={n_neighbors}")
        return self.knn

    def train_naive_bayes(
        self, X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0
    ) -> "MultinomialNB":
        """Train Naive Bayes classifier"""
        self.naive_bayes = MultinomialNB(alpha=alpha)
        self.naive_bayes.fit(X_train, y_train)
        print(f"Naive Bayes trained with alpha={alpha}")
        return self.naive_bayes

    def train_random_forest(
        self, X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 100
    ) -> "RandomForestClassifier":
        """Train Random Forest classifier"""
        self.random_forest = RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=-1, random_state=42
        )
        self.random_forest.fit(X_train, y_train)
        print(f"Random Forest trained with {n_estimators} trees")
        return self.random_forest

    def train_xgboost(
        self, X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 200
    ) -> "XGBClassifier":
        """Train XGBoost classifier"""
        self.xgboost = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self.xgboost.fit(X_train, y_train)
        print(f"XGBoost trained with {n_estimators} estimators")
        return self.xgboost

    def evaluate(
        self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of model for logging

        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred
        )

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_prob) if len(np.unique(y_prob)) > 1 else 0.0,
        }

        print(f"\n{model_name} Evaluation:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

        return metrics

    def save_model(self, model, filename: str, model_dir: str = "results/models"):
        """Save trained model to disk"""
        model_path = Path(model_dir) / filename
        model_path.parent.mkdir(parents=True, exist_ok=True)
        dump(model, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, filename: str, model_dir: str = "results/models"):
        """Load trained model from disk"""
        model_path = Path(model_dir) / filename
        return load(model_path)

    def save_vectorizer(self, filename: str = "tfidf_vectorizer.joblib", model_dir: str = "results/models"):
        """Save TF-IDF vectorizer"""
        self.save_model(self.tfidf, filename, model_dir)

    def save_svd(self, filename: str = "svd_reducer.joblib", model_dir: str = "results/models"):
        """Save SVD reducer"""
        if self.svd is not None:
            self.save_model(self.svd, filename, model_dir)


def get_classical_model(
    model_name: str, model_dir: str = "results/models"
) -> Any:
    """
    Load pre-trained classical model

    Args:
        model_name: Name of model file (e.g., 'knn_model.joblib')
        model_dir: Directory containing models

    Returns:
        Loaded model object
    """
    model_path = Path(model_dir) / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return load(model_path)


def get_tfidf_vectorizer(model_dir: str = "results/models") -> "TfidfVectorizer":
    """Load pre-trained TF-IDF vectorizer"""
    return get_classical_model("tfidf_vectorizer.joblib", model_dir)


def get_svd_reducer(model_dir: str = "results/models") -> "TruncatedSVD":
    """Load pre-trained SVD reducer"""
    return get_classical_model("svd_reducer.joblib", model_dir)
