from src.py.ml_core.data_loader import EnhancedDataLoader
import pytest
import pandas as pd
import numpy as np
from src.py.ml_core.volatility_regime import EnhancedVolatilityDetector

@pytest.fixture
def sample_data():
    dates = pd.date_range(end=pd.Timestamp.today(), periods=200, freq='min')
    return pd.DataFrame({
        'open': np.linspace(98, 102, 200),
        'high': np.linspace(102, 106, 200),
        'low': np.linspace(97, 101, 200),
        'close': np.linspace(100, 104, 200),
        'volume': np.random.poisson(10000, 200),
        'vwap': (np.random.normal(100, 5, 200)) / 3,
        'mid_price': (np.random.normal(100, 5, 200) + np.random.normal(100, 5, 200)) / 2,
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
    data = pd.DataFrame({
        'days_since_dividend': [3, 100],
        'split_ratio': [1.0, 2.0],
        'close': [100.0, 200.0],
        'open': [99.5, 198.0],
        'high': [102.0, 204.0],  # Introduce price range for volatility
        'low': [98.0, 196.0],
        'volume': [10000, 20000],
        'vwap': [(101+98+100)/3, (202.5+197.5+200)/3],
        'mid_price': [(99.5+100)/2, (198.0+200)/2],
        'bid_ask_spread': [0.1, 0.5]
    })
    
    detector = EnhancedVolatilityDetector()
    enhanced = detector._calculate_event_volatility(data)
    
    # Use numpy's nan-safe comparison functions
    assert np.greater(enhanced.loc[0, 'event_volatility'], 
                     enhanced.loc[1, 'event_volatility'])
    
    # Validate NaN handling
    assert not np.isnan(enhanced['event_volatility']).any()   

def test_sequence_generation(sample_data):
    loader = EnhancedDataLoader()
    sequences = loader.create_sequences(sample_data)
    
    # Validate LSTM input dimensions
    assert sequences.shape[0] == 140  # 200 - 60 window
    assert sequences.shape[2] == len(loader.feature_columns)
