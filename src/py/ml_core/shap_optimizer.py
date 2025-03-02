# src/py/ml_core/shap_optimizer.py
import shap
import tensorflow as tf
import numpy as np
import joblib
from tqdm import tqdm
from .data_loader import EnhancedDataLoader

class EnhancedSHAPOptimizer:
    # def __init__(self, model_path='s3://quant-trader-data-gintoki/models/regime_model.h5',
    #              background_samples=1000):
    
    #test line
    def __init__(self, model_path: str = 'src/py/ml_core/models/regime_model.h5', background_samples=1):
        self.model = tf.keras.models.load_model(model_path)
        self.input_name = self.model.layers[0].name  # Get actual input name
        self.data_loader = EnhancedDataLoader()
        self.background = self._load_production_background(background_samples)


        # CRUCIAL: Use concrete model inputs/outputs
        model_input = self.model.inputs[0]
        model_output = self.model.outputs[0]

        self.explainer = shap.GradientExplainer(
            model=self.model,
            data=self.background,
            batch_size=32
        )
        
        # self.explainer = shap.DeepExplainer(
        #     (model_input, model_output),  # Input/output tuple
        #     self.background
        # )
        

        # self.explainer = shap.DeepExplainer(
        #     (self.model.get_layer(self.input_name).input, self.model.output),
        #     self.background
        # )        
        self.essential_features = ['days_since_dividend', 'split_ratio', 'bid_ask_spread']

    def _load_production_background(self, n_samples):
        """Load real market data from S3"""
        # df = self.data_loader.load_ticker_data('AMZN')
        # sequences = self.data_loader.create_sequences(df)
        # return sequences[-n_samples:]
        
        # Test line
        return np.random.randn(n_samples, 60, 20)  # Match production shape

    def calculate_shap(self, data):
        # GradientExplainer handles batches internally
        return self.explainer.shap_values(
            data, 
            nsamples=200,  # Only valid parameters stay
            # nsamples=1000  # production - adjust based on accuracy needs 
        )


    # def calculate_shap(self, X: np.ndarray) -> np.ndarray:
    #     """Compute SHAP values with GPU acceleration"""
    #     shap_values = []
    #     batch_size = 100
        
    #     with tf.device('/GPU:0'):
    #         for i in tqdm(range(0, len(X), batch_size)):
    #             batch = X[i:i+batch_size]
    #             shap_values.append(self.explainer.shap_values(batch))
                
    #     return np.concatenate(shap_values)

    # def optimize_features(self, data_path: str, top_k: int = 15) -> np.ndarray:
    def optimize_features(self, input_data, top_k=10):
        # data = joblib.load(data_path)
        """input_data: np array (n_samples, 60, 20) OR file path string"""
        if isinstance(input_data, str):
            data = joblib.load(input_data)
        else:
            data = input_data  # Treat as pre-loaded
        """SHAP-based feature optimization with essential retention"""
        sample = data[np.random.choice(len(data), 2000, replace=False)]
        
        shap_vals = self.calculate_shap(sample)
        importance = np.abs(shap_vals).mean((0, 1))
        
        # Force include essential features
        essential_idx = [self.data_loader.feature_columns.index(f) 
                        for f in self.essential_features]
        importance[essential_idx] += 1000
        
        return np.argsort(importance)[-top_k:]
