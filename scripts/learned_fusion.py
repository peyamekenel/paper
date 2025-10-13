"""
Learned Fusion - ML-based combination of similarity signals.

Instead of using a fixed alpha parameter, train a model to learn
the optimal combination of BERT and TF-IDF similarities.
"""
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os


class LearnedFusionModel:
    """
    ML model that learns to combine BERT and TF-IDF similarities.
    
    Uses metadata overlap as supervision signal to learn optimal fusion.
    """
    
    def __init__(self, model_type: str = "gbm"):
        """
        Initialize fusion model.
        
        Args:
            model_type: Type of ML model to use
                - "gbm": Gradient Boosting (default, best performance)
                - "rf": Random Forest (faster, good performance)
                - "ridge": Ridge Regression (fastest, linear)
                - "lasso": Lasso Regression (feature selection)
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = ["bert_sim", "tfidf_sim", "bert_squared", "tfidf_squared", "interaction"]
        
        # Initialize model
        if model_type == "gbm":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        elif model_type == "rf":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "ridge":
            self.model = Ridge(alpha=1.0)
        elif model_type == "lasso":
            self.model = Lasso(alpha=0.01)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def create_training_data(
        self,
        bert_sim: np.ndarray,
        tfidf_sim: np.ndarray,
        df: pd.DataFrame,
        sample_size: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training data using metadata overlap as supervision.
        
        The idea: Movies with high metadata overlap (genres, keywords)
        should have high similarity scores.
        
        Args:
            bert_sim: BERT similarity matrix (N x N)
            tfidf_sim: TF-IDF similarity matrix (N x N)
            df: DataFrame with movie metadata
            sample_size: Number of movie pairs to sample (not movies!)
        
        Returns:
            Tuple of (X_features, y_labels)
        """
        n_movies = len(df)
        
        # Sample random movie pairs directly
        np.random.seed(42)
        
        X_features = []
        y_labels = []
        
        # Sample random pairs efficiently
        for _ in range(sample_size):
            # Sample two different movies
            idx_i, idx_j = np.random.choice(n_movies, size=2, replace=False)
            
            # Feature vector: [bert_sim, tfidf_sim, bert^2, tfidf^2, bert*tfidf]
            bert_score = bert_sim[idx_i, idx_j]
            tfidf_score = tfidf_sim[idx_i, idx_j]
            
            features = [
                bert_score,
                tfidf_score,
                bert_score ** 2,
                tfidf_score ** 2,
                bert_score * tfidf_score
            ]
            
            # Label: metadata overlap score (simplified to use tags)
            tags1 = set(str(df.iloc[idx_i].get("tags", "")).split())
            tags2 = set(str(df.iloc[idx_j].get("tags", "")).split())
            
            if len(tags1) > 0 and len(tags2) > 0:
                intersection = len(tags1 & tags2)
                union = len(tags1 | tags2)
                label = intersection / union if union > 0 else 0.0
            else:
                label = 0.0
            
            X_features.append(features)
            y_labels.append(label)
        
        return np.array(X_features), np.array(y_labels)
    
    def _compute_metadata_similarity(self, movie1: pd.Series, movie2: pd.Series) -> float:
        """
        Compute metadata similarity between two movies.
        
        Uses genre overlap, keyword overlap, and other metadata features.
        """
        import ast
        
        def parse_json_field(field):
            """Parse JSON-like string field."""
            if pd.isna(field):
                return set()
            if isinstance(field, list):
                return set([d.get("name", "") for d in field if isinstance(d, dict)])
            try:
                parsed = ast.literal_eval(str(field))
                if isinstance(parsed, list):
                    return set([d.get("name", "") for d in parsed if isinstance(d, dict)])
            except:
                pass
            return set()
        
        # Get metadata from DataFrame (from original data if available)
        # Since we only have preprocessed data, use tags as proxy
        tags1 = set(str(movie1.get("tags", "")).split())
        tags2 = set(str(movie2.get("tags", "")).split())
        
        if len(tags1) == 0 or len(tags2) == 0:
            return 0.0
        
        # Jaccard similarity
        intersection = len(tags1 & tags2)
        union = len(tags1 | tags2)
        
        if union == 0:
            return 0.0
        
        jaccard = intersection / union
        
        # Scale to [0, 1] range with some noise for better training
        similarity = min(1.0, jaccard * 2.0)  # Scale up a bit
        
        return similarity
    
    def train(
        self,
        bert_sim: np.ndarray,
        tfidf_sim: np.ndarray,
        df: pd.DataFrame,
        sample_size: int = 10000,
        verbose: bool = True
    ) -> dict:
        """
        Train the fusion model.
        
        Args:
            bert_sim: BERT similarity matrix
            tfidf_sim: TF-IDF similarity matrix
            df: DataFrame with metadata
            sample_size: Number of training samples
            verbose: Print training progress
        
        Returns:
            Dictionary with training metrics
        """
        if verbose:
            print(f"Creating training data (sampling {sample_size} pairs)...")
        
        X, y = self.create_training_data(bert_sim, tfidf_sim, df, sample_size)
        
        if verbose:
            print(f"  Training samples: {len(X)}")
            print(f"  Label range: [{y.min():.3f}, {y.max():.3f}]")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        if verbose:
            print(f"Training {self.model_type} model...")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        metrics = {
            "train_mse": float(train_mse),
            "test_mse": float(test_mse),
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "n_samples": len(X),
            "model_type": self.model_type
        }
        
        if verbose:
            print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
            print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if verbose:
                print(f"  Feature importances:")
                for name, imp in zip(self.feature_names, importances):
                    print(f"    {name:15s}: {imp:.4f}")
            metrics["feature_importances"] = {
                name: float(imp) for name, imp in zip(self.feature_names, importances)
            }
        
        return metrics
    
    def predict_similarity(
        self,
        bert_sim: np.ndarray,
        tfidf_sim: np.ndarray
    ) -> np.ndarray:
        """
        Predict fused similarity using trained model.
        
        Args:
            bert_sim: BERT similarity matrix (N x N)
            tfidf_sim: TF-IDF similarity matrix (N x N)
        
        Returns:
            Fused similarity matrix (N x N)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        n = bert_sim.shape[0]
        fused_sim = np.zeros((n, n), dtype=np.float32)
        
        # Predict for all pairs
        for i in range(n):
            # Create features for row i
            bert_scores = bert_sim[i, :]
            tfidf_scores = tfidf_sim[i, :]
            
            features = np.column_stack([
                bert_scores,
                tfidf_scores,
                bert_scores ** 2,
                tfidf_scores ** 2,
                bert_scores * tfidf_scores
            ])
            
            # Predict
            predictions = self.model.predict(features)
            fused_sim[i, :] = predictions
        
        # Ensure symmetry
        fused_sim = (fused_sim + fused_sim.T) / 2
        
        # Clip to [0, 1]
        fused_sim = np.clip(fused_sim, 0, 1)
        
        return fused_sim
    
    def save(self, path: str) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        joblib.dump({
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self.feature_names
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load trained model from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        data = joblib.load(path)
        self.model = data["model"]
        self.model_type = data["model_type"]
        self.feature_names = data["feature_names"]
        print(f"Model loaded from {path}")


def create_learned_similarity(
    bert_sim: np.ndarray,
    tfidf_sim: np.ndarray,
    df: pd.DataFrame,
    model_type: str = "gbm",
    sample_size: int = 10000,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Create learned fusion similarity matrix.
    
    Convenience function that trains and applies learned fusion in one step.
    
    Args:
        bert_sim: BERT similarity matrix
        tfidf_sim: TF-IDF similarity matrix
        df: DataFrame with metadata
        model_type: ML model type ("gbm", "rf", "ridge", "lasso")
        sample_size: Training sample size
        save_path: Optional path to save trained model
        verbose: Print progress
    
    Returns:
        Tuple of (fused_similarity_matrix, training_metrics)
    """
    # Initialize and train model
    fusion_model = LearnedFusionModel(model_type=model_type)
    metrics = fusion_model.train(bert_sim, tfidf_sim, df, sample_size, verbose)
    
    # Generate fused similarity matrix
    if verbose:
        print("Generating fused similarity matrix...")
    fused_sim = fusion_model.predict_similarity(bert_sim, tfidf_sim)
    
    # Save if requested
    if save_path:
        fusion_model.save(save_path)
    
    return fused_sim, metrics

