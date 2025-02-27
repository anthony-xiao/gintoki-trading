import pytest
import numpy as np
from ml_core.shap_optimizer import EnhancedSHAPOptimizer

def test_feature_retention():
    optimizer = EnhancedSHAPOptimizer()
    dummy_data = np.random.randn(1000, 20)
    top_features = optimizer.optimize_features(dummy_data, top_k=5)
    
    assert len(top_features) == 5
    assert all(0 <= idx < 20 for idx in top_features)
