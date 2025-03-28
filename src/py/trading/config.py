"""
Trading configuration parameters
"""
import os
from .utils import get_latest_model_version

# S3 configuration
S3_BUCKET = 'quant-trader-data-gintoki'
MODEL_PREFIX = 'models/'

def get_model_paths():
    """Get the latest model paths from S3"""
    paths = {}
    
    # Get latest versions for each model
    model_types = ['volatility', 'transformer', 'transformer_optimized']
    for model_type in model_types:
        prefix = f"{MODEL_PREFIX}{model_type}/models/"
        latest_version = get_latest_model_version(S3_BUCKET, prefix)
        if latest_version:
            paths[f'{model_type}_model_path'] = f's3://{S3_BUCKET}/{latest_version}'
    
    # Get latest ensemble config
    ensemble_prefix = f"{MODEL_PREFIX}ensemble/models/"
    latest_ensemble = get_latest_model_version(S3_BUCKET, ensemble_prefix, extension='.json')
    if latest_ensemble:
        paths['ensemble_config_path'] = f's3://{S3_BUCKET}/{latest_ensemble}'
    
    return paths

# Get latest model paths
MODEL_PATHS = get_model_paths()

DEFAULT_CONFIG = {
    # Trading parameters
    'initial_capital': 100000.0,
    'risk_per_trade': 0.02,  # 2% risk per trade
    'max_position_size': 0.1,  # 10% of portfolio
    'min_position_size': 0.01,  # 1% of portfolio
    'min_signal_threshold': 0.1,  # Minimum signal strength to trade
    
    # Paper trading
    'paper_trading': True,
    
    # Model paths (dynamically updated)
    **MODEL_PATHS,
    
    # S3 configuration
    's3_bucket': S3_BUCKET,
    'model_prefix': MODEL_PREFIX,
    
    # Risk management
    'max_drawdown': 0.15,  # 15% maximum drawdown
    'stop_loss': 0.05,  # 5% stop loss
    'take_profit': 0.15,  # 15% take profit
    
    # Performance tracking
    'performance_update_interval': 60,  # seconds
    'trade_history_size': 1000,  # number of trades to keep in history
    
    # Logging
    'log_level': 'INFO',
    'log_file': 'trading.log'
} 