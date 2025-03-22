import argparse
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict
import os
from .trading_engine import TradingEngine
from ..ml_core.data_loader import EnhancedDataLoader

# Configure logging
def setup_logging():
    logger = logging.getLogger("trading")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    
    # File handler
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    file_handler = logging.FileHandler(
        f"{log_dir}/trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s"
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logging()

def load_config() -> Dict:
    """Load trading configuration"""
    return {
        'initial_capital': 100000.0,
        'risk_per_trade': 0.02,  # 2% risk per trade
        'min_signal_threshold': 0.1,
        'max_position_size': 0.1,  # 10% max position
        'min_position_size': 0.01,  # 1% min position
        'seq_length': 60,
        'd_model': 64,
        'num_heads': 8,
        'volatility_model_path': 'src/py/ml_core/models/regime_model.h5',
        'transformer_model_path': 'src/py/ml_core/models/transformer_trend_optimized.h5',
        'risk_management': {
            'max_vol': 0.015,
            'max_spread': 0.002
        },
        'regime_weights': {
            'high_volatility': {'transformer': 0.6, 'xgb': 0.3, 'lstm': 0.1},
            'low_volatility': {'transformer': 0.3, 'xgb': 0.5, 'lstm': 0.2},
            'neutral': {'transformer': 0.4, 'xgb': 0.4, 'lstm': 0.2}
        }
    }

def main():
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description='Run algorithmic trading system')
        parser.add_argument('--ticker', type=str, default='SMCI',
                          help='Ticker symbol to trade')
        parser.add_argument('--initial-capital', type=float, default=100000.0,
                          help='Initial trading capital')
        parser.add_argument('--risk-per-trade', type=float, default=0.02,
                          help='Risk per trade as percentage')
        args = parser.parse_args()
        
        # Load configuration
        config = load_config()
        config.update({
            'initial_capital': args.initial_capital,
            'risk_per_trade': args.risk_per_trade
        })
        
        # Initialize components
        logger.info("Initializing trading system...")
        engine = TradingEngine(config)
        data_loader = EnhancedDataLoader()
        
        # Initialize models
        logger.info("Loading models...")
        engine.initialize()
        
        # Main trading loop
        logger.info(f"Starting trading for {args.ticker}")
        while True:
            try:
                # Get latest market data
                data = data_loader.load_ticker_data(args.ticker)
                if data is None or data.empty:
                    logger.warning("No market data available")
                    time.sleep(60)  # Wait 1 minute before retrying
                    continue
                
                # Process market data
                signal_data = engine.process_market_data(data)
                
                # Get current price
                current_price = data['close'].iloc[-1]
                
                # Execute trade if conditions are met
                trade = engine.execute_trade(signal_data, current_price)
                if trade:
                    logger.info(f"Executed trade: {trade}")
                    
                    # Update performance
                    performance = engine.update_performance(current_price)
                    logger.info(f"Trade performance: {performance}")
                    
                    # Log portfolio summary
                    portfolio = engine.get_portfolio_summary()
                    logger.info(f"Portfolio summary: {portfolio}")
                
                # Wait for next iteration
                time.sleep(60)  # 1-minute intervals
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(60)  # Wait before retrying
                continue
                
    except KeyboardInterrupt:
        logger.info("Trading system stopped by user")
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 