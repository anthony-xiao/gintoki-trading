import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from datetime import datetime, timedelta
from ..ml_core.performance import EnhancedPerformanceAnalyzer
from backtesting import Strategy, Backtest

logger = logging.getLogger(__name__)

class PerformanceTracker:
    def __init__(self, config: Dict):
        """Initialize performance tracker"""
        self.config = config
        self.trade_history: List[Dict] = []
        self.daily_returns: List[Dict] = []
        self.performance_metrics: Dict = {}
        self.analyzer = EnhancedPerformanceAnalyzer(None)  # Will be set when we have data
        
    def add_trade(self, trade: Dict) -> None:
        """Add a trade to history"""
        self.trade_history.append(trade)
        
        # Update daily returns
        trade_date = trade['timestamp'].date()
        if not self.daily_returns or self.daily_returns[-1]['date'] != trade_date:
            self.daily_returns.append({
                'date': trade_date,
                'return': 0.0,
                'trades': []
            })
        
        self.daily_returns[-1]['trades'].append(trade)
        
    def update_metrics(self, current_positions: Dict[str, float], 
                      current_prices: Dict[str, float]) -> Dict:
        """Update performance metrics"""
        try:
            # Calculate current P&L
            total_pnl = 0.0
            position_metrics = {}
            
            for symbol, qty in current_positions.items():
                if symbol in current_prices:
                    price = current_prices[symbol]
                    position = self.get_position(symbol)
                    if position:
                        pnl = qty * (price - position['avg_entry_price'])
                        total_pnl += pnl
                        position_metrics[symbol] = {
                            'pnl': pnl,
                            'return': pnl / (qty * position['avg_entry_price']),
                            'qty': qty,
                            'price': price
                        }
            
            # Calculate daily return
            daily_return = total_pnl / self.get_portfolio_value()
            
            # Update metrics
            self.performance_metrics.update({
                'total_pnl': total_pnl,
                'daily_return': daily_return,
                'positions': position_metrics,
                'timestamp': datetime.now(),
                'portfolio_value': self.get_portfolio_value(),
                'sharpe_ratio': self.calculate_sharpe_ratio(),
                'max_drawdown': self.calculate_max_drawdown(),
                'win_rate': self.calculate_win_rate()
            })
            
            logger.info(f"Updated performance metrics: {self.performance_metrics}")
            return self.performance_metrics
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {str(e)}")
            raise
            
    def get_position(self, symbol: str) -> Dict:
        """Get current position details from trade history"""
        try:
            # Find the most recent trade for this symbol
            symbol_trades = [t for t in self.trade_history if t['symbol'] == symbol]
            if not symbol_trades:
                return None
                
            latest_trade = max(symbol_trades, key=lambda x: x['timestamp'])
            return {
                'avg_entry_price': latest_trade['price'],
                'qty': latest_trade['position_size']
            }
            
        except Exception as e:
            logger.error(f"Failed to get position: {str(e)}")
            return None
            
    def get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        try:
            # Start with cash
            total_value = self.config.get('initial_capital', 100000.0)
            
            # Add value of all positions
            for trade in self.trade_history:
                if trade['order_status'] == 'filled':
                    total_value += trade['value']
                    
            return total_value
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio value: {str(e)}")
            return self.config.get('initial_capital', 100000.0)
            
    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not self.daily_returns:
                return 0.0
                
            returns = pd.Series([d['return'] for d in self.daily_returns])
            if len(returns) < 2:
                return 0.0
                
            return np.sqrt(252) * (returns.mean() / returns.std())
            
        except Exception as e:
            logger.error(f"Failed to calculate Sharpe ratio: {str(e)}")
            return 0.0
            
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if not self.daily_returns:
                return 0.0
                
            portfolio_values = []
            current_value = self.config.get('initial_capital', 100000.0)
            
            for daily in self.daily_returns:
                current_value *= (1 + daily['return'])
                portfolio_values.append(current_value)
                
            portfolio_values = pd.Series(portfolio_values)
            rolling_max = portfolio_values.expanding().max()
            drawdowns = (portfolio_values - rolling_max) / rolling_max
            
            return abs(drawdowns.min())
            
        except Exception as e:
            logger.error(f"Failed to calculate max drawdown: {str(e)}")
            return 0.0
            
    def calculate_win_rate(self) -> float:
        """Calculate win rate"""
        try:
            if not self.trade_history:
                return 0.0
                
            winning_trades = [t for t in self.trade_history 
                            if t['order_status'] == 'filled' and t['pnl'] > 0]
            return len(winning_trades) / len(self.trade_history)
            
        except Exception as e:
            logger.error(f"Failed to calculate win rate: {str(e)}")
            return 0.0
            
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        return {
            'metrics': self.performance_metrics,
            'trade_history': self.trade_history,
            'daily_returns': self.daily_returns,
            'summary': {
                'total_trades': len(self.trade_history),
                'winning_trades': len([t for t in self.trade_history if t.get('pnl', 0) > 0]),
                'total_pnl': self.performance_metrics.get('total_pnl', 0.0),
                'sharpe_ratio': self.performance_metrics.get('sharpe_ratio', 0.0),
                'max_drawdown': self.performance_metrics.get('max_drawdown', 0.0),
                'win_rate': self.performance_metrics.get('win_rate', 0.0)
            }
        } 