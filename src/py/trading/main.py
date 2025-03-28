import os
import logging
import time
from datetime import datetime
from typing import List, Dict
import pandas as pd

from .trading_engine import TradingEngine
from .config import DEFAULT_CONFIG
from ..ml_core.data_loader import EnhancedDataLoader
from ..data_engineering.polygon_realtime import PolygonRealtime

# Configure logging
logging.basicConfig(
    level=DEFAULT_CONFIG['log_level'],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DEFAULT_CONFIG['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingSystem:
    def __init__(self, symbols: List[str], config: Dict = None):
        """Initialize the trading system"""
        self.symbols = symbols
        self.config = config or DEFAULT_CONFIG
        
        # Initialize components
        self.trading_engine = TradingEngine(self.config)
        self.data_loader = EnhancedDataLoader()
        self.polygon = PolygonRealtime()
        
    def initialize(self):
        """Initialize all components"""
        try:
            # Initialize trading engine
            self.trading_engine.initialize()
            
            # Connect to Polygon
            self.polygon.authenticate()
            
            # Subscribe to symbols
            for symbol in self.symbols:
                self.polygon.subscribe(symbol)
                
            logger.info("✅ Trading system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading system: {str(e)}")
            raise
            
    def process_market_data(self, data: pd.DataFrame, symbol: str):
        """Process incoming market data"""
        try:
            # Generate trading signals
            signal_data = self.trading_engine.process_market_data(data)
            
            # Execute trade if signal is strong enough
            if signal_data['position_size'] > 0:
                current_price = data['close'].iloc[-1]
                trade = self.trading_engine.execute_trade(
                    signal_data, 
                    current_price,
                    symbol
                )
                
                if trade:
                    logger.info(f"Executed trade: {trade}")
                    
            # Update performance metrics
            performance = self.trading_engine.update_performance(
                data['close'].iloc[-1],
                symbol
            )
            
            if performance:
                logger.info(f"Performance update: {performance}")
                
        except Exception as e:
            logger.error(f"Failed to process market data: {str(e)}")
            
    def run(self):
        """Main trading loop"""
        try:
            self.initialize()
            
            while True:
                # Process market data for each symbol
                for symbol in self.symbols:
                    data = self.polygon.get_latest_data(symbol)
                    if data is not None:
                        self.process_market_data(data, symbol)
                        
                # Get portfolio summary
                portfolio = self.trading_engine.get_portfolio_summary()
                logger.info(f"Portfolio summary: {portfolio}")
                
                # Sleep for update interval
                time.sleep(self.config['performance_update_interval'])
                
        except KeyboardInterrupt:
            logger.info("Shutting down trading system...")
        except Exception as e:
            logger.error(f"Trading system error: {str(e)}")
            raise

def main():
    """Main entry point"""
    # Trading symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Create and run trading system
    trading_system = TradingSystem(symbols)
    trading_system.run()

if __name__ == "__main__":
    main() 