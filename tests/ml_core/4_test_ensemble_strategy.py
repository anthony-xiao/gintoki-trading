import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.py.ml_core.ensemble_strategy import AdaptiveEnsembleTrader

# Mock feature mask indices aligned with test data columns
MOCK_FEATURE_MASK = np.array([0, 1, 4, 5])  # Positions of: days_since_dividend, split_ratio, div_alert, split_alert

@pytest.fixture(autouse=True)
def mock_feature_mask(tmp_path):
    mask_file = tmp_path / "feature_mask.npz"
    np.savez(mask_file, mask=MOCK_FEATURE_MASK)
    return mask_file

@pytest.fixture
def mock_trader():
    with patch('tensorflow.keras.models.load_model'), \
         patch('numpy.load') as mock_npload:
        
        # Mock feature mask load
        mock_npload.return_value = {'mask': MOCK_FEATURE_MASK}
        
        # Mock classifiers
        mock_xgb = MagicMock()
        mock_xgb.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        mock_rf = MagicMock()
        mock_rf.predict_proba.return_value = np.array([[0.4, 0.6]])
        
        trader = AdaptiveEnsembleTrader()
        trader.models = {
            'xgb': mock_xgb,
            'rf': mock_rf
        }
        return trader

@pytest.fixture
def sample_features():
    """Test data with ALL required feature columns"""
    return pd.DataFrame({
        'days_since_dividend': [2, 3650],  # Used for div_alert
        'split_ratio': [1.0, 2.0],         # Used for split_alert
        'bid_ask_spread': [0.1, 0.3],
        'close': [100, 200],
        'div_alert': [1, 0],               # Added missing column
        'split_alert': [0, 1]              # Added missing column
    }).iloc[:, MOCK_FEATURE_MASK]

def test_dividend_boost(mock_trader, sample_features):
    mock_trader.models['xgb'].predict_proba.return_value = np.array([[0.2, 0.8]])
    mock_trader.models['rf'].predict_proba.return_value = np.array([[0.3, 0.7]])
    
    mock_features = np.array([[0.2, 0.8, 1, 0.3]])
    signal = mock_trader._mean_reversion_signal(
        sample_features.iloc[[0]], 
        mock_features
    )
    assert signal == 0.9375  # (0.8 + 0.7)/2 * 1.25

def test_split_boost(mock_trader, sample_features):
    mock_trader.models['xgb'].predict_proba.return_value = np.array([[0.1, 0.9]])
    mock_trader.models['rf'].predict_proba.return_value = np.array([[0.15, 0.85]])
    
    mock_features = np.array([[0.9, 0.85, 0, 0.15]])
    signal = mock_trader._momentum_signal(
        sample_features.iloc[[1]], 
        mock_features
    )
    assert signal == 1.215  # max(0.9, 0.85) * 1.35

def test_spread_adjustment(mock_trader):
    df = pd.DataFrame({'spread_ratio': [1.5]})
    adjustment = mock_trader._spread_adjustment_factor(df)
    assert adjustment == 0.5  # 1.2 - 1.5 clipped to 0.5
