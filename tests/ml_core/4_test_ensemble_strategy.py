import pytest
import pandas as pd
import numpy as np
from src.py.ml_core.ensemble_strategy import EnhancedEnsembleTrader
from unittest.mock import MagicMock, patch
import h5py
from typing import List  # For type hints
import os  # For path handlingf

@pytest.fixture
def sample_features():
    return pd.DataFrame({
        'days_since_dividend': [2, 3650],
        'split_ratio': [1.0, 2.0],
        'bid_ask_spread': [0.1, 0.3],
        'close': [100, 200]
    })

@pytest.fixture
def mock_trader():
    trader = EnhancedEnsembleTrader()
    
    # Mock model loading
    with patch('tensorflow.keras.models.load_model') as mock_load:
        mock_load.return_value = MagicMock()
        trader.models['lstm'] = mock_load.return_value
        
    trader.models['lstm'].predict.return_value = [[0.8]]
    return trader

def test_dividend_boost(mock_trader, sample_features):
    signal = mock_trader._mean_reversion_signal(sample_features.iloc[[0]])
    assert signal > 0.5

def test_split_boost(mock_trader, sample_features):
    signal = mock_trader._momentum_signal(sample_features.iloc[[1]])
    assert signal > 0.5

def test_spread_adjustment(mock_trader):
    adjustment = mock_trader._spread_adjustment_factor(pd.Series({'spread_ratio': 1.5}))
    assert adjustment == 0.7

