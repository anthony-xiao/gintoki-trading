# tests/ml_core/test_full_pipeline.py
from src.py.ml_core.shap_optimizer import EnhancedSHAPOptimizer
from src.py.ml_core.transformer_trend import TransformerTrendAnalyzer
from src.py.ml_core.volatility_regime import EnhancedVolatilityDetector
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch, ANY
from pathlib import Path
from src.py.ml_core.data_loader import EnhancedDataLoader
from src.py.ml_core.ensemble_strategy import AdaptiveEnsembleTrader

@pytest.fixture
def mock_trading_data():
    dates = pd.date_range(end=pd.Timestamp.now(), periods=200, freq='1min')
    return pd.DataFrame({
        'open': np.random.normal(100, 5, 200),
        'high': np.random.normal(105, 5, 200),
        'low': np.random.normal(95, 5, 200),
        'close': np.random.normal(100, 5, 200),
        'volume': np.random.poisson(10000, 200),
        'vwap': np.random.normal(100, 5, 200),
        'days_since_dividend': np.random.randint(0, 100, 200),
        'split_ratio': np.random.choice([1.0, 2.0], 200),
        'bid_ask_spread': np.random.uniform(0.1, 0.5, 200),
        'mid_price': np.random.normal(100, 5, 200),
        'symbol': 'TEST'
    }, index=dates)

@pytest.fixture(autouse=True)
def mock_model_files():
    """Create dummy model files"""
    model_dir = Path("src/py/ml_core/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "mock_vol.h5").touch()
    (model_dir / "mock_trans.keras").touch()
    yield
    # Cleanup
    (model_dir / "mock_vol.h5").unlink(missing_ok=True)
    (model_dir / "mock_trans.keras").unlink(missing_ok=True)

def test_full_pipeline_with_transformer(mock_trading_data):
    """Test complete pipeline from S3 data to trading signals"""
    # Configure test instance
    config = {
        'volatility_model_path': 'mock.h5',
        'transformer_model_path': 'mock.keras',
        'risk_management': {'max_vol': 0.02, 'max_spread': 0.005},
        'regime_weights': {
            'high_volatility': {'transformer': 0.6, 'xgb': 0.3, 'lstm': 0.1},
            'low_volatility': {'transformer': 0.3, 'xgb': 0.5, 'lstm': 0.2},
            'neutral': {'transformer': 0.4, 'xgb': 0.4, 'lstm': 0.2}
        },
        'feature_columns': [
            'open', 'high', 'low', 'close', 'volume', 'vwap',
            'days_since_dividend', 'split_ratio', 'bid_ask_spread', 'mid_price'
        ]
    }

    """Test complete pipeline from S3 data to trading signals"""
    mock_volatility_detector = MagicMock(spec=EnhancedVolatilityDetector)
    mock_volatility_detector.train = Mock()
    mock_volatility_detector.create_labels = Mock(return_value=mock_trading_data)

    # Mock Keras models
    mock_lstm = MagicMock()
    mock_lstm.predict.return_value = np.array([[[0.8, 0.1, 0.1]]])
    
    mock_transformer = MagicMock()
    mock_transformer.predict.return_value = np.array([[0.85]])

    # with patch.multiple(
    #     'src.py.ml_core.data_loader.EnhancedDataLoader',
    #     load_ticker_data=Mock(return_value=mock_trading_data)
    # ), \
    # patch.multiple(
    #     'src.py.ml_core.volatility_regime.EnhancedVolatilityDetector',
    #     train=Mock(),
    #     create_labels=Mock(return_value=mock_trading_data)
    # ), \
    # patch.multiple(
    #     'src.py.ml_core.shap_optimizer.EnhancedSHAPOptimizer',
    #     optimize_features=mock_shap.optimize_features
    # ), \
    # patch.multiple(
    #     'src.py.ml_core.transformer_trend.TransformerTrendAnalyzer',
    #     train=Mock(),
    #     predict_trend_strength=mock_transformer.predict
    # ):

    with patch(
        'src.py.ml_core.volatility_regime.EnhancedVolatilityDetector',
        return_value=mock_volatility_detector
    ), \
    patch(
        'src.py.ml_core.transformer_trend.TransformerTrendAnalyzer',
        autospec=True
    ) as mock_transformer:
        mock_transformer.return_value.predict_trend_strength.return_value = np.array([0.85])

    
        # Initialize pipeline
        loader = EnhancedDataLoader()
        volatility_detector = EnhancedVolatilityDetector()
        shap_optimizer = EnhancedSHAPOptimizer()
        transformer = TransformerTrendAnalyzer()
        ensemble = AdaptiveEnsembleTrader(config)

        # Execute pipeline
        data = loader.load_ticker_data('TEST')
        # volatility_detector.train(['TEST'])
        # shap_features = shap_optimizer.optimize_features(data)
        # transformer.train('mock_data.npz')
        # signals = ensemble.calculate_signals(data)
        signals = ensemble.calculate_signals(mock_trading_data)


        # Validate outputs
        assert -1 <= signals['signal'] <= 1
        assert signals['regime'] == 'high_volatility'
        assert signals['components']['transformer'] > 0.5
        assert signals['confidence'] > 0.6

def test_risk_adjustment_with_transformer():
    """Test risk management integration with transformer signals"""
    config = {
        'volatility_model_path': 'mock.h5',
        'transformer_model_path': 'mock.keras',
        'risk_management': {'max_vol': 0.02, 'max_spread': 0.005},
        'regime_weights': {
            'high_volatility': {'transformer': 0.6, 'xgb': 0.3, 'lstm': 0.1},
            'low_volatility': {'transformer': 0.3, 'xgb': 0.5, 'lstm': 0.2},
            'neutral': {'transformer': 0.4, 'xgb': 0.4, 'lstm': 0.2}
        },
        'feature_columns': []
    }

    test_cases = [
        {'vol': 0.03, 'spread': 0.006, 'transformer_signal': 0.8, 'expected': 0.48},
        {'vol': 0.02, 'spread': 0.004, 'transformer_signal': 0.6, 'expected': 0.72},
        {'vol': 0.01, 'spread': 0.002, 'transformer_signal': 0.9, 'expected': 1.35}
    ]

    with patch(
        'tensorflow.keras.models.load_model',
        return_value=MagicMock()
    ):
        for case in test_cases:
            ensemble = AdaptiveEnsembleTrader(config)
            ensemble.models['transformer_trend'].predict = Mock(return_value=[[case['transformer_signal']]])
            
            mock_data = pd.DataFrame({
                'volatility': [case['vol']],
                'spread_ratio': [case['spread']],
                **{col: [0] for col in config['feature_columns']}
            })
            
            signals = ensemble.calculate_signals(mock_data)
            adjusted_signal = signals['signal'] * signals['confidence']
            assert abs(adjusted_signal - case['expected']) < 0.1