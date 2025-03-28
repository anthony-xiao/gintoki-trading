import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime, timedelta
from ..ml_core.model_factory import ModelFactory
from ..ml_core.data_loader import EnhancedDataLoader
from .alpaca_client import AlpacaTradingClient
from .performance_tracker import PerformanceTracker
from alpaca.trading.enums import OrderSide

logger = logging.getLogger(__name__)

class TradingEngine:
    def __init__(self, config: Dict):
        """Initialize the trading engine with configuration"""
        self.config = config
        self.model_factory = ModelFactory(config)
        self.data_loader = EnhancedDataLoader()
        self.positions: Dict[str, float] = {}  # Current positions
        self.trade_history: List[Dict] = []    # Trade history
        self.cash = config.get('initial_capital', 100000.0)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% risk per trade
        
        # Initialize Alpaca client
        self.alpaca = AlpacaTradingClient(paper=config.get('paper_trading', True))
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker(config)
        
    def initialize(self) -> None:
        """Initialize models and load historical data"""
        try:
            # Log model paths being used
            logger.info("Initializing trading engine with the following model versions:")
            for model_type in ['volatility', 'transformer', 'transformer_optimized']:
                model_path = self.config.get(f'{model_type}_model_path')
                logger.info(f"  {model_type}: {model_path}")
            logger.info(f"  ensemble config: {self.config.get('ensemble_config_path')}")
            
            # Create and load models
            self.model_factory.create_models()
            self.model_factory.load_models({
                'volatility': self.config.get('volatility_model_path'),
                'transformer': self.config.get('transformer_model_path'),
                'transformer_optimized': self.config.get('transformer_optimized_model_path'),
                'ensemble': self.config.get('ensemble_config_path')
            })
            
            # Get initial account state
            account_summary = self.alpaca.get_account_summary()
            self.cash = account_summary['cash']
            
            # Get current positions
            for position in account_summary['positions']:
                self.positions[position['symbol']] = position['qty']
            
            logger.info("✅ Trading engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading engine: {str(e)}")
            raise
            
    def process_market_data(self, data: pd.DataFrame) -> Dict:
        """Process market data and generate trading signals"""
        try:
            # Get trading signal from ensemble
            signal, metadata = self.model_factory.get_trading_signal(data)
            
            # Calculate position size based on risk
            position_size = self._calculate_position_size(signal, data)
            
            return {
                'signal': signal,
                'position_size': position_size,
                'metadata': metadata,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to process market data: {str(e)}")
            raise
            
    def execute_trade(self, signal_data: Dict, current_price: float, symbol: str) -> Optional[Dict]:
        """Execute trade based on signal and current market conditions"""
        try:
            signal = signal_data['signal']
            position_size = signal_data['position_size']
            
            # Skip if signal is too weak
            if abs(signal) < self.config.get('min_signal_threshold', 0.1):
                return None
                
            # Calculate trade details
            trade_value = position_size * current_price
            if trade_value > self.cash:
                logger.warning("Insufficient funds for trade")
                return None
                
            # Determine order side based on signal
            side = OrderSide.BUY if signal > 0 else OrderSide.SELL
            
            # Execute order through Alpaca
            order_result = self.alpaca.execute_order(symbol, position_size, side)
            if not order_result:
                logger.error("Failed to execute order")
                return None
                
            # Record trade
            trade = {
                'timestamp': signal_data['timestamp'],
                'symbol': symbol,
                'signal': signal,
                'position_size': position_size,
                'price': current_price,
                'value': trade_value,
                'metadata': signal_data['metadata'],
                'order_id': order_result['id'],
                'order_status': order_result['status']
            }
            
            # Update positions and cash
            self.positions[symbol] = position_size
            self.cash -= trade_value
            self.trade_history.append(trade)
            
            # Add trade to performance tracker
            self.performance_tracker.add_trade(trade)
            
            logger.info(f"Executed trade: {trade}")
            return trade
            
        except Exception as e:
            logger.error(f"Failed to execute trade: {str(e)}")
            raise
            
    def _calculate_position_size(self, signal: float, data: pd.DataFrame) -> float:
        """Calculate position size based on risk management rules"""
        try:
            # Get volatility from data
            volatility = data['volatility'].iloc[-1]
            
            # Calculate base position size (risk-adjusted)
            risk_amount = self.cash * self.risk_per_trade
            base_size = risk_amount / (volatility * abs(signal))
            
            # Apply position limits
            max_position = self.cash * self.config.get('max_position_size', 0.1)  # 10% max
            min_position = self.cash * self.config.get('min_position_size', 0.01)  # 1% min
            
            # Scale by signal strength
            position_size = base_size * abs(signal)
            
            # Apply limits
            position_size = np.clip(position_size, min_position, max_position)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Failed to calculate position size: {str(e)}")
            raise
            
    def update_performance(self, current_price: float, symbol: str) -> Dict:
        """Update performance metrics and adjust model weights"""
        try:
            if not self.positions:
                return {}
                
            # Get current position from Alpaca
            position = self.alpaca.get_position(symbol)
            if not position:
                return {}
                
            # Update performance metrics
            current_prices = {symbol: current_price}
            performance = self.performance_tracker.update_metrics(
                self.positions,
                current_prices
            )
            
            # Update model weights based on performance
            self.model_factory.update_ensemble_weights({
                'transformer': performance['daily_return'],
                'lstm': performance['daily_return'],
                'xgb': performance['daily_return']
            })
            
            return performance
            
        except Exception as e:
            logger.error(f"Failed to update performance: {str(e)}")
            raise
            
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        account_summary = self.alpaca.get_account_summary()
        performance_summary = self.performance_tracker.get_performance_summary()
        
        return {
            **account_summary,
            'performance': performance_summary
        } 