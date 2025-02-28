import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from moto import mock_aws
import boto3
from src.py.ml_core.data_loader import EnhancedDataLoader

@pytest.fixture
def mock_s3():
    with mock_aws():
        s3 = boto3.client('s3')
        s3.create_bucket(Bucket='quant-trader-data-gintoki')
        yield s3

@pytest.fixture
def sample_data():
    dates = pd.date_range(end=datetime.today(), periods=100, freq='T')
    return pd.DataFrame({
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(100, 200, 100),
        'low': np.random.uniform(100, 200, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.randint(1000, 10000, 100),
        'symbol': 'AMZN'
    }, index=dates)

def test_load_ticker_data(mock_s3, sample_data):
    # Upload test data
    mock_s3.put_object(
        Bucket='quant-trader-data-gintoki',
        Key='historical/AMZN/aggregates/20230101.parquet',
        Body=sample_data.to_parquet()
    )
    
    # Test corporate action merging
    corporate_actions = pd.DataFrame({
        'symbol': ['AMZN'],
        'ex_date': [datetime.today() - timedelta(days=2)],
        'type': ['dividend'],
        'payment_date': [datetime.today()],
        'ratio': [1.0]
    })
    mock_s3.put_object(
        Bucket='quant-trader-data-gintoki',
        Key='corporate_actions/corporate_actions.parquet',
        Body=corporate_actions.to_parquet()
    )
    
    loader = EnhancedDataLoader()
    data = loader.load_ticker_data('AMZN')
    
    # Validate core features
    assert {'open', 'high', 'low', 'close', 'volume'}.issubset(data.columns)
    assert 'days_since_dividend' in data.columns
    assert 'bid_ask_spread' in data.columns
    assert data['days_since_dividend'].min() == 2  # From corporate action fixture

def test_corporate_action_edge_cases(mock_s3, sample_data):
    # Test no corporate actions scenario
    loader = EnhancedDataLoader()
    data = loader.load_ticker_data('AMZN')
    assert data['days_since_dividend'].iloc[0] == 3650
    assert data['split_ratio'].iloc[0] == 1.0

def test_quote_processing(mock_s3):
    # Upload quote data
    quotes = pd.DataFrame({
        'bid_price': [100, 101, 102],
        'ask_price': [101, 102, 103],
        'bid_size': [500, 600, 700],
        'ask_size': [500, 600, 700]
    }, index=pd.date_range(end=datetime.today(), periods=3, freq='T'))
    
    mock_s3.put_object(
        Bucket='quant-trader-data-gintoki',
        Key='historical/AMZN/quotes/20230101.parquet',
        Body=quotes.to_parquet()
    )
    
    loader = EnhancedDataLoader()
    data = loader.load_ticker_data('AMZN')
    
    # Validate spread calculations
    assert np.allclose(data['bid_ask_spread'], 1.0)
    assert np.allclose(data['mid_price'], [100.5, 101.5, 102.5])
