import pytest
import numpy as np
from src.py.ml_core.shap_optimizer import EnhancedSHAPOptimizer

def test_feature_retention(monkeypatch):
    # Mock background data loader
    def mock_background(self, n_samples):
        return np.random.randn(n_samples, 60, 20)
    
    monkeypatch.setattr(EnhancedSHAPOptimizer, '_load_production_background', mock_background)
    
    # Initialize with valid 3D data
    optimizer = EnhancedSHAPOptimizer()
    dummy_data = np.random.randn(2000, 60, 20)  # Matches sample_size
    
    # Test feature optimization
    top_features = optimizer.optimize_features(input_data=dummy_data, top_k=5)
    
    # Validate results
    assert len(top_features) == 5
    assert all(0 <= idx < 20 for idx in top_features)

