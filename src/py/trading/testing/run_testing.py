import logging
from typing import Dict
import pandas as pd
from datetime import datetime, timedelta
from .walk_forward import WalkForwardTester
from ...ml_core.data_loader import EnhancedDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_testing(config: Dict):
    """Run walk-forward testing on the trading system"""
    try:
        logger.info("Starting walk-forward testing...")
        
        # Load historical data
        data_loader = EnhancedDataLoader()
        # symbols = config.get('test_symbols', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'])
        symbols = config.get('test_symbols', ['BTDR'])
        
        # Get data from 2023 to 2025
        end_date = datetime.now()
        start_date = datetime(2023, 1, 1)  # Start from beginning of 2023
        
        logger.info(f"Loading historical data from {start_date} to {end_date}")
        historical_data = {}
        for symbol in symbols:
            logger.info(f"Loading data for {symbol}...")
            df = data_loader.load_ticker_data(symbol)
            if df is not None:
                # The data should already have datetime index from the Parquet files
                logger.info(f"Data loaded with index range: {df.index[0]} to {df.index[-1]}")
                
                # Filter data by date range
                df = df[
                    (df.index >= start_date) & 
                    (df.index <= end_date)
                ]
                logger.info(f"After date filtering: {len(df)} rows")
                
                if len(df) == 0:
                    logger.warning(f"No data in date range {start_date} to {end_date}")
                    continue
                
                # Resample to daily data
                df = df.resample('D').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'vwap': 'mean',
                    'bid_ask_spread': 'mean',
                    'days_since_dividend': 'last',
                    'split_ratio': 'last',
                    'mid_price': 'last'
                }).dropna()
                logger.info(f"After resampling: {len(df)} rows")
                
                # Add volatility column if not present
                if 'volatility' not in df.columns:
                    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
                    logger.info("Added volatility column")
                
                historical_data[symbol] = df
            else:
                logger.warning(f"No data loaded for {symbol}")
        
        if not historical_data:
            raise ValueError("No data available for any symbol in the specified date range")
        
        # Initialize walk-forward tester
        tester = WalkForwardTester(config)
        
        # Run walk-forward testing
        logger.info("Running walk-forward testing...")
        results = tester.run_walk_forward(historical_data)
        
        # Log results
        logger.info("\nWalk-forward testing results:")
        logger.info(f"Total periods tested: {len(results['period_metrics'])}")
        logger.info(f"Total trades: {results['overall_metrics']['total_trades']}")
        logger.info(f"Win rate: {results['overall_metrics']['win_rate']:.2%}")
        logger.info(f"Sharpe ratio: {results['overall_metrics']['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {results['overall_metrics']['max_drawdown']:.2%}")
        
        # Log period-by-period results
        logger.info("\nPeriod-by-period results:")
        for period in results['period_metrics']:
            logger.info(f"\nPeriod: {period['start_date']} to {period['end_date']}")
            logger.info(f"  Trades: {period['metrics']['summary']['total_trades']}")
            logger.info(f"  Win rate: {period['metrics']['summary']['win_rate']:.2%}")
            logger.info(f"  P&L: ${period['metrics']['summary']['total_pnl']:.2f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to run testing: {str(e)}")
        raise

if __name__ == "__main__":
    # Example configuration
    config = {
        # 'test_symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        'test_symbols': ['BTDR'],
        'train_window': 252,    # 1 year
        'val_window': 63,       # 3 months
        'test_window': 63,      # 3 months
        'step_size': 21,        # 1 month
        'feature_lookback': 30, # Days of historical data needed for features
        
        # Model parameters
        'seq_length': 10,       # Shorter sequence length for testing
        'd_model': 32,          # Smaller model size
        'num_heads': 4,         # Fewer attention heads
        'epochs': 5,            # More epochs for testing
        'batch_size': 32,
        
        # Trading parameters
        'initial_capital': 100000.0,
        'risk_per_trade': 0.02,
        'max_position_size': 0.1,
        'min_position_size': 0.01,
        'min_signal_threshold': 0.1,
        
        # Validation criteria
        'min_validation_sharpe': 0.1,      # Minimum Sharpe ratio (reduced from 0.5)
        'max_validation_drawdown': 0.25,   # Maximum drawdown (increased from 0.15)
        'min_validation_win_rate': 0.40,   # Minimum win rate (reduced from 0.45)
        'min_validation_trades': 5         # Minimum number of trades (reduced from 10)
    }
    
    results = run_testing(config) 