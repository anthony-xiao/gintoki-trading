from src.py.ml_core.data_loader import EnhancedDataLoader
import pytest
import pandas as pd
import numpy as np
from src.py.ml_core.volatility_regime import EnhancedVolatilityDetector

@pytest.fixture
def sample_data():
    dates = pd.date_range(end=pd.Timestamp.today(), periods=200, freq='T')
    return pd.DataFrame({
        'open': np.random.normal(100, 5, 200),
        'high': np.random.normal(105, 5, 200),
        'low': np.random.normal(95, 5, 200),
        'close': np.random.normal(100, 5, 200),
        'volume': np.random.poisson(10000, 200),
        'days_since_dividend': [3650]*200,
        'split_ratio': [1.0]*200,
        'bid_ask_spread': np.random.uniform(0.1, 0.5, 200)
    }, index=dates)

def test_regime_labeling(sample_data):
    detector = EnhancedVolatilityDetector()
    labeled = detector.create_labels(sample_data.copy())
    
    # Validate regime classes exist
    assert 'regime' in labeled.columns
    assert set(labeled['regime']) <= {0, 1, 2}
    
    # Validate volatility calculation
    assert 'event_volatility' in labeled.columns
    assert labeled['event_volatility'].min() > 0

def test_volatility_boosting():
    # Test corporate action volatility boosts
    data = pd.DataFrame({
        'days_since_dividend': [3, 100],
        'split_ratio': [1.0, 2.0],
        'close': [100, 200]
    })
    detector = EnhancedVolatilityDetector()
    enhanced = detector._calculate_event_volatility(data)
    
    assert enhanced.loc[0, 'event_volatility'] > enhanced.loc[1, 'event_volatility']
    assert enhanced.loc[1, 'event_volatility'] > 0

def test_sequence_generation(sample_data):
    loader = EnhancedDataLoader()
    sequences = loader.create_sequences(sample_data)
    
    # Validate LSTM input dimensions
    assert sequences.shape[0] == 140  # 200 - 60 window
    assert sequences.shape[2] == len(loader.feature_columns)
