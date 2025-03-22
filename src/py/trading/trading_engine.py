import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime, timedelta
from ..ml_core.model_factory import ModelFactory
from ..ml_core.data_loader import EnhancedDataLoader

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
        
    def initialize(self) -> None:
        """Initialize models and load historical data"""
        try:
            # Create and load models
            self.model_factory.create_models()
            self.model_factory.load_models({
                'volatility': self.config.get('volatility_model_path'),
                'transformer': self.config.get('transformer_model_path')
            })
            
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
            
    def execute_trade(self, signal_data: Dict, current_price: float) -> Optional[Dict]:
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
                
            # Record trade
            trade = {
                'timestamp': signal_data['timestamp'],
                'signal': signal,
                'position_size': position_size,
                'price': current_price,
                'value': trade_value,
                'metadata': signal_data['metadata']
            }
            
            # Update positions and cash
            self.positions['current'] = position_size
            self.cash -= trade_value
            self.trade_history.append(trade)
            
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
            
    def update_performance(self, current_price: float) -> Dict:
        """Update performance metrics and adjust model weights"""
        try:
            if not self.positions:
                return {}
                
            # Calculate current P&L
            position = self.positions['current']
            pnl = position * (current_price - self.trade_history[-1]['price'])
            
            # Calculate performance metrics
            performance = {
                'pnl': pnl,
                'return': pnl / self.trade_history[-1]['value'],
                'timestamp': datetime.now()
            }
            
            # Update model weights based on performance
            self.model_factory.update_ensemble_weights({
                'transformer': performance['return'],
                'lstm': performance['return'],
                'xgb': performance['return']
            })
            
            return performance
            
        except Exception as e:
            logger.error(f"Failed to update performance: {str(e)}")
            raise
            
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        return {
            'cash': self.cash,
            'positions': self.positions,
            'total_trades': len(self.trade_history),
            'current_value': self.cash + sum(
                pos * self.trade_history[-1]['price'] 
                for pos in self.positions.values()
            )
        } 