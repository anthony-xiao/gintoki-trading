import shap
import numpy as np
import joblib
from tqdm import tqdm

class FeatureOptimizer:
    def __init__(self, model_path='regime_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.explainer = shap.DeepExplainer(self.model)
    
    def analyze(self, X_sample):
        return self.explainer.shap_values(X_sample)
    
    def optimize_features(self, data_path, top_k=20):
        X = np.load(data_path)['X_train']
        sample = X[np.random.choice(X.shape[0], 1000, replace=False)]
        
        # Distributed SHAP computation
        shaps = []
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            for i in tqdm(range(0, len(sample), 100)):
                shaps.append(self.analyze(sample[i:i+100]))
        
        shaps = np.concatenate(shaps)
        importance = np.abs(shaps).mean(0).sum(1)
        top_features = np.argsort(importance)[-top_k:]
        
        # Save feature mask
        np.savez('feature_mask.npz', mask=top_features)
        return top_features
