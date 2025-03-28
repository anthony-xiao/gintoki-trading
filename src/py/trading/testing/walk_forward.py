import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
from ...ml_core.model_factory import ModelFactory
from ...ml_core.data_loader import EnhancedDataLoader
from ..performance_tracker import PerformanceTracker
from backtesting import Strategy, Backtest

logger = logging.getLogger(__name__)

class MLStrategy(Strategy):
    """Basic ML-based trading strategy"""
    def init(self):
        self.signal = self.I(lambda: 0.0, name='signal')
        self.position_size = self.I(lambda: 0.0, name='position_size')
        
    def next(self):
        if self.signal[-1] > 0.1:  # Buy signal
            self.buy(size=self.position_size[-1])
        elif self.signal[-1] < -0.1:  # Sell signal
            self.sell(size=self.position_size[-1])

class WalkForwardTester:
    def __init__(self, config: Dict):
        """Initialize walk-forward testing system"""
        self.config = config
        self.data_loader = EnhancedDataLoader()
        self.model_factory = ModelFactory(config)
        self.performance_tracker = PerformanceTracker(config)
        
        # Testing parameters
        self.train_window = config.get('train_window', 252)    # 1 year of trading days
        self.val_window = config.get('val_window', 63)         # 3 months of trading days
        self.test_window = config.get('test_window', 63)       # 3 months of trading days
        self.step_size = config.get('step_size', 21)           # 1 month of trading days
        
        # Data preprocessing parameters
        self.feature_lookback = config.get('feature_lookback', 30)  # Days of historical data needed for features
        
    def run_walk_forward(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Run walk-forward testing with proper data separation"""
        try:
            results = []
            
            # Get the first symbol's data for testing
            symbol = list(data.keys())[0]
            symbol_data = data[symbol]
            total_days = len(symbol_data)
            
            logger.info(f"Running walk-forward testing on {symbol} with {total_days} days of data")
            
            # Calculate number of iterations
            current_start = 0
            while current_start + self.train_window + self.val_window + self.test_window <= total_days:
                # Split data into train, validation, and test periods
                train_end = current_start + self.train_window
                val_end = train_end + self.val_window
                test_end = val_end + self.test_window
                
                # Get data for each period
                train_data = symbol_data.iloc[current_start:train_end]
                val_data = symbol_data.iloc[train_end:val_end]
                test_data = symbol_data.iloc[val_end:test_end]
                
                logger.info(f"Running walk-forward iteration:")
                logger.info(f"  Train period: {train_data.index[0]} to {train_data.index[-1]}")
                logger.info(f"  Validation period: {val_data.index[0]} to {val_data.index[-1]}")
                logger.info(f"  Test period: {test_data.index[0]} to {test_data.index[-1]}")
                
                # Train models on training data
                self.model_factory.create_models()
                self.model_factory.train_models(train_data)
                
                # Validate models on validation data
                val_metrics = self._validate_period(val_data)
                
                # Only proceed with testing if validation metrics meet criteria
                if self._check_validation_metrics(val_metrics):
                    # Test on test period
                    iteration_results = self._test_period(test_data)
                    results.append(iteration_results)
                else:
                    logger.warning(f"Validation metrics did not meet criteria for period {val_data.index[0]} to {val_data.index[-1]}")
                
                # Move forward
                current_start += self.step_size
            
            # Aggregate results
            summary = self._aggregate_results(results)
            logger.info(f"Walk-forward testing completed. Summary: {summary}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to run walk-forward testing: {str(e)}")
            raise
            
    def _validate_period(self, val_data: pd.DataFrame) -> Dict:
        """Validate models on validation period"""
        try:
            val_results = []
            
            # Process each day in the validation period
            for date, day_data in val_data.groupby(level=0):
                # Get trading signal
                signal, metadata = self.model_factory.get_trading_signal(day_data)
                
                # Calculate position size
                position_size = self._calculate_position_size(signal, day_data)
                
                # Record results
                result = {
                    'date': date,
                    'signal': signal,
                    'position_size': position_size,
                    'price': day_data['close'].iloc[-1],
                    'metadata': metadata
                }
                
                val_results.append(result)
            
            # Calculate validation metrics
            val_metrics = {
                'sharpe_ratio': self._calculate_sharpe_ratio(val_results),
                'max_drawdown': self._calculate_max_drawdown(val_results),
                'win_rate': self._calculate_win_rate(val_results),
                'total_trades': len(val_results),
                'avg_position_size': np.mean([r['position_size'] for r in val_results])
            }
            
            return val_metrics
            
        except Exception as e:
            logger.error(f"Failed to validate period: {str(e)}")
            raise
            
    def _check_validation_metrics(self, val_metrics: Dict) -> bool:
        """Check if validation metrics meet criteria"""
        try:
            # Define validation criteria
            criteria = {
                'min_sharpe': self.config.get('min_validation_sharpe', 0.5),
                'max_drawdown': self.config.get('max_validation_drawdown', 0.15),
                'min_win_rate': self.config.get('min_validation_win_rate', 0.45),
                'min_trades': self.config.get('min_validation_trades', 10)
            }
            
            # Check each criterion
            if val_metrics['sharpe_ratio'] < criteria['min_sharpe']:
                logger.warning(f"Sharpe ratio {val_metrics['sharpe_ratio']:.2f} below minimum {criteria['min_sharpe']}")
                return False
                
            if val_metrics['max_drawdown'] > criteria['max_drawdown']:
                logger.warning(f"Max drawdown {val_metrics['max_drawdown']:.2%} above maximum {criteria['max_drawdown']:.2%}")
                return False
                
            if val_metrics['win_rate'] < criteria['min_win_rate']:
                logger.warning(f"Win rate {val_metrics['win_rate']:.2%} below minimum {criteria['min_win_rate']:.2%}")
                return False
                
            if val_metrics['total_trades'] < criteria['min_trades']:
                logger.warning(f"Total trades {val_metrics['total_trades']} below minimum {criteria['min_trades']}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to check validation metrics: {str(e)}")
            return False
            
    def _calculate_sharpe_ratio(self, results: List[Dict]) -> float:
        """Calculate Sharpe ratio for a period"""
        try:
            if not results:
                return 0.0
                
            returns = []
            for i in range(1, len(results)):
                prev_price = results[i-1]['price']
                curr_price = results[i]['price']
                returns.append((curr_price - prev_price) / prev_price)
                
            if not returns:
                return 0.0
                
            returns = pd.Series(returns)
            return np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() != 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate Sharpe ratio: {str(e)}")
            return 0.0
            
    def _calculate_max_drawdown(self, results: List[Dict]) -> float:
        """Calculate maximum drawdown for a period"""
        try:
            if not results:
                return 0.0
                
            prices = pd.Series([r['price'] for r in results])
            rolling_max = prices.expanding().max()
            drawdowns = (prices - rolling_max) / rolling_max
            
            return abs(drawdowns.min()) if not drawdowns.empty else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate max drawdown: {str(e)}")
            return 0.0
            
    def _calculate_win_rate(self, results: List[Dict]) -> float:
        """Calculate win rate for a period"""
        try:
            if not results:
                return 0.0
                
            winning_trades = len([r for r in results if r['signal'] > 0 and r['price'] > r['price']])
            return winning_trades / len(results)
            
        except Exception as e:
            logger.error(f"Failed to calculate win rate: {str(e)}")
            return 0.0
            
    def _test_period(self, test_data: pd.DataFrame) -> Dict:
        """Test models on a specific period"""
        try:
            period_results = []
            
            # Process each day in the test period
            for date, day_data in test_data.groupby(level=0):
                # Get trading signal
                signal, metadata = self.model_factory.get_trading_signal(day_data)
                
                # Calculate position size
                position_size = self._calculate_position_size(signal, day_data)
                
                # Record results
                result = {
                    'date': date,
                    'signal': signal,
                    'position_size': position_size,
                    'price': day_data['close'].iloc[-1],
                    'metadata': metadata
                }
                
                # Update performance metrics
                self.performance_tracker.update_metrics(
                    {'current': position_size},
                    {'current': day_data['close'].iloc[-1]}
                )
                
                period_results.append(result)
            
            # Get performance metrics for this period
            period_metrics = self.performance_tracker.get_performance_summary()
            
            return {
                'period_results': period_results,
                'metrics': period_metrics,
                'start_date': test_data.index[0],
                'end_date': test_data.index[-1]
            }
            
        except Exception as e:
            logger.error(f"Failed to test period: {str(e)}")
            raise
            
    def _calculate_position_size(self, signal: float, data: pd.DataFrame) -> float:
        """Calculate position size based on risk management rules"""
        try:
            # Get volatility from data
            volatility = data['volatility'].iloc[-1]
            
            # Calculate base position size (risk-adjusted)
            risk_amount = self.config.get('initial_capital', 100000.0) * self.config.get('risk_per_trade', 0.02)
            base_size = risk_amount / (volatility * abs(signal))
            
            # Apply position limits
            max_position = self.config.get('initial_capital', 100000.0) * self.config.get('max_position_size', 0.1)
            min_position = self.config.get('initial_capital', 100000.0) * self.config.get('min_position_size', 0.01)
            
            # Scale by signal strength
            position_size = base_size * abs(signal)
            
            # Apply limits
            position_size = np.clip(position_size, min_position, max_position)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Failed to calculate position size: {str(e)}")
            raise
            
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results from all walk-forward iterations"""
        try:
            # Combine all period results
            all_results = []
            for result in results:
                all_results.extend(result['period_results'])
            
            # Calculate overall metrics
            overall_metrics = {
                'total_trades': len(all_results),
                'winning_trades': len([r for r in all_results if r['signal'] > 0 and r['price'] > r['price']]),
                'losing_trades': len([r for r in all_results if r['signal'] > 0 and r['price'] < r['price']]),
                'average_position_size': np.mean([r['position_size'] for r in all_results]),
                'average_signal': np.mean([r['signal'] for r in all_results]),
                'sharpe_ratio': self.performance_tracker.calculate_sharpe_ratio(),
                'max_drawdown': self.performance_tracker.calculate_max_drawdown(),
                'win_rate': self.performance_tracker.calculate_win_rate()
            }
            
            # Calculate metrics by period
            period_metrics = []
            for result in results:
                period_metrics.append({
                    'start_date': result['start_date'],
                    'end_date': result['end_date'],
                    'metrics': result['metrics']
                })
            
            return {
                'overall_metrics': overall_metrics,
                'period_metrics': period_metrics,
                'all_results': all_results
            }
            
        except Exception as e:
            logger.error(f"Failed to aggregate results: {str(e)}")
            raise 