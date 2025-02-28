from src.py.ml_core.data_loader import EnhancedDataLoader
import shap
import numpy as np
import joblib
from tqdm import tqdm
import tensorflow as tf

class EnhancedSHAPOptimizer:
    def __init__(self, model_path: str = 'regime_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.explainer = shap.DeepExplainer(
            self.model,
            np.zeros((1, 60, len(EnhancedDataLoader().feature_columns)))
        )
        
    def calculate_shap(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values with GPU acceleration"""
        shap_values = []
        batch_size = 100
        
        with tf.device('/GPU:0'):
            for i in tqdm(range(0, len(X), batch_size)):
                batch = X[i:i+batch_size]
                shap_values.append(self.explainer.shap_values(batch))
                
        return np.concatenate(shap_values)

    def optimize_features(self, data_path: str, top_k: int = 15) -> np.ndarray:
        """Feature selection with enhanced criteria"""
        data = joblib.load(data_path)
        sample = data[np.random.choice(len(data), 2000, replace=False)]
        
        # Compute SHAP importance
        shap_vals = self.calculate_shap(sample)
        importance = np.abs(shap_vals).mean((0, 1))
        
        # Enforce corporate action retention
        essential = ['days_since_dividend', 'split_ratio', 'bid_ask_spread']
        essential_idx = [EnhancedDataLoader().feature_columns.index(f) for f in essential]
        importance[essential_idx] += 1000  # Force inclusion
        
        return np.argsort(importance)[-top_k:]
