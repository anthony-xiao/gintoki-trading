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
        # Create client with explicit region
        s3 = boto3.client('s3', region_name='us-west-2')
        # Create bucket with location constraint
        s3.create_bucket(
            Bucket='quant-trader-data-gintoki',
            CreateBucketConfiguration={'LocationConstraint': 'us-west-2'}
        )
        yield s3

@pytest.fixture
def sample_data():
    dates = pd.date_range(end=datetime.today(), periods=100, freq='min')
    return pd.DataFrame({
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(100, 200, 100),
        'low': np.random.uniform(100, 200, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.randint(1000, 10000, 100),
        'symbol': 'AMZN'
    }, index=dates)

def test_load_ticker_data(mock_s3, sample_data):
    # Upload test data with proper key formatting
    mock_s3.put_object(
        Bucket='quant-trader-data-gintoki',
        Key='historical/AMZN/aggregates/day/2023-01-01_to_2025-02-26.parquet',
        Body=sample_data.to_parquet()
    )
    
    # Upload dividend data with correct path
    dividend_data = pd.DataFrame({
        'symbol': ['AMZN'],
        'ex_date': [datetime.today() - timedelta(days=2)],
        'payment_date': [datetime.today()],
        'amount': [0.50]
    })
    mock_s3.put_object(
        Bucket='quant-trader-data-gintoki',
        Key='corporate_actions/dividends/2023-01-01_to_2025-02-26.parquet',
        Body=dividend_data.to_parquet()
    )
    
    # Upload split data with correct path
    split_data = pd.DataFrame({
        'symbol': ['AMZN'],
        'ex_date': [datetime.today() - timedelta(days=5)],
        'ratio': [4.0]
    })
    mock_s3.put_object(
        Bucket='quant-trader-data-gintoki',
        Key='corporate_actions/splits/2023-01-01_to_2025-02-26.parquet',
        Body=split_data.to_parquet()
    )
    
    loader = EnhancedDataLoader()
    data = loader.load_ticker_data('AMZN')
    
    # Validate corporate action features exist
    assert 'days_since_dividend' in data.columns
    assert 'split_ratio' in data.columns
    
    # Validate core features
    assert not data[['open', 'high', 'low', 'close', 'volume']].isnull().any().any()
    
    # Verify corporate action processing
    assert data['days_since_dividend'].min() <= 2
    assert data['split_ratio'].max() == 4.0

def test_corporate_action_edge_cases(mock_s3, sample_data):
    # Upload data without corporate actions
    mock_s3.put_object(
        Bucket='quant-trader-data-gintoki',
        Key='historical/AMZN/aggregates/day/2023-01-01_to_2025-02-26.parquet',
        Body=sample_data.to_parquet()
    )
    
    loader = EnhancedDataLoader()
    data = loader.load_ticker_data('AMZN')
    
    # Validate default values
    assert data['days_since_dividend'].iloc[0] == 3650
    assert data['split_ratio'].iloc[0] == 1.0

def test_quote_processing(mock_s3):
    # Upload quote data with proper formatting
    quotes = pd.DataFrame({
        'bid_price': [100.0, 101.0, 102.0],
        'ask_price': [101.0, 102.0, 103.0],
        'bid_size': [500, 600, 700],
        'ask_size': [500, 600, 700]
    }, index=pd.date_range(end=datetime.today(), periods=3, freq='min'))
    
    mock_s3.put_object(
        Bucket='quant-trader-data-gintoki',
        Key='historical/AMZN/quotes/2025-02-26.parquet',
        Body=quotes.to_parquet()
    )
    
    loader = EnhancedDataLoader()
    data = loader.load_ticker_data('AMZN')
    
    # Validate quote-derived features
    assert 'bid_ask_spread' in data.columns
    assert 'mid_price' in data.columns
    assert np.allclose(data['bid_ask_spread'].dropna(), 1.0)
