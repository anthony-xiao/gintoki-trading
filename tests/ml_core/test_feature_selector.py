import pytest
from unittest.mock import Mock, patch
import numpy as np
from src.py.ml_core.ensemble_strategy import AdaptiveEnsembleTrader
from pytest import ImportError

class TestFeatureSelectorInitialization:
    def test_valid_selector_configuration(self):
        """Test valid feature selector configurations"""
        trader = AdaptiveEnsembleTrader(
            feature_selector_config={
                'module': 'sklearn.feature_selection',
                'class': 'SelectFromModel',
                'params': {'estimator': 'RandomForestClassifier(n_estimators=50)'}
            }
        )
        assert hasattr(trader, 'feature_selector'), "Selector not initialized"
        assert trader.feature_selector is not None

    def test_invalid_selector_module(self):
        """Test invalid selector module handling"""
        with pytest.raises(ImportError):
            AdaptiveEnsembleTrader(
                feature_selector_config={
                    'module': 'invalid.module',
                    'class': 'FakeSelector'
                }
            )

    def test_invalid_selector_class(self):
        """Test invalid selector class handling"""
        with pytest.raises(AttributeError):
            AdaptiveEnsembleTrader(
                feature_selector_config={
                    'module': 'sklearn.feature_selection',
                    'class': 'NonExistentSelector'
                }
            )

    def test_selector_integration(self):
        """Test selector integration with training workflow"""
        trader = AdaptiveEnsembleTrader()
        dummy_X = np.random.rand(100, 10)
        dummy_y = np.random.randint(0, 2, 100)
        
        # Test fit transform integration
        trader.feature_selector.fit(dummy_X, dummy_y)
        transformed_X = trader.feature_selector.transform(dummy_X)
        assert transformed_X.shape[1] &lt;= dummy_X.shape[1], "Feature selection failed"

        # Verify selector used in model
        assert hasattr(trader.models[0], 'feature_importances_'), "Model not using selected features"

