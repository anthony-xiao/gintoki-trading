import pytest
from src.py.ml_core.data_loader import EnhancedDataLoader
from src.py.ml_core.volatility_regime import EnhancedVolatilityDetector
from src.py.ml_core.ensemble_strategy import EnhancedEnsembleTrader

def test_full_pipeline():
    # Load sample data
    loader = EnhancedDataLoader()
    data = loader.load_ticker_data('AMZN')
    
    # Train models
    detector = EnhancedVolatilityDetector()
    detector.train(['AMZN'], epochs=1)  # Smoke test
    
    trader = EnhancedEnsembleTrader()
    trader.train(['AMZN'])
    
    # Generate signal
    signal = trader.predict(data.iloc[-100:])
    assert -1 <= signal <= 1
