import pytest
import pandas as pd
import numpy as np
from src.py.ml_core.ensemble_strategy import EnhancedEnsembleTrader

@pytest.fixture
def sample_features():
    return pd.DataFrame({
        'days_since_dividend': [2, 3650],
        'split_ratio': [1.0, 2.0],
        'bid_ask_spread': [0.1, 0.3],
        'close': [100, 200]
    })

def test_dividend_boost(sample_features):
    trader = EnhancedEnsembleTrader()
    signal = trader._mean_reversion_signal(sample_features.iloc[[0]])
    assert signal > 0.5  # Boosted signal

def test_split_boost(sample_features):
    trader = EnhancedEnsembleTrader()
    signal = trader._momentum_signal(sample_features.iloc[[1]])
    assert signal > 0.5  # Boosted signal

def test_spread_adjustment():
    trader = EnhancedEnsembleTrader()
    adjustment = trader._spread_adjustment_factor(pd.Series({'spread_ratio': 1.5}))
    assert adjustment == 0.7  # 1.2 - 1.5 = 0.7 clipped to 0.5-1.0
