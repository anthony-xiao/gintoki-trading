import pandas as pd
from backtesting import Backtest, Strategy

class EnhancedPerformanceAnalyzer:
    def __init__(self, strategy_class=None):
        self.strategy_class = strategy_class
        
    def analyze_enhancements(self, data):
        if self.strategy_class is None:
            return pd.DataFrame({
                'Metric': ['Return', 'Sharpe', 'Max Drawdown'],
                'Base': [0.0, 0.0, 0.0],
                'Enhanced': [0.0, 0.0, 0.0]
            }).set_index('Metric')
            
        # Test base vs enhanced
        bt = Backtest(data.query("days_since_dividend > 30 and split_ratio == 1.0"), 
                     self.strategy_class)
        base_stats = bt.run()
        
        bt = Backtest(data, self.strategy_class)
        enhanced_stats = bt.run()
        
        return pd.DataFrame({
            'Metric': ['Return', 'Sharpe', 'Max Drawdown'],
            'Base': [
                base_stats['Return [%]'],
                base_stats['Sharpe Ratio'],
                base_stats['Max. Drawdown [%]']
            ],
            'Enhanced': [
                enhanced_stats['Return [%]'], 
                enhanced_stats['Sharpe Ratio'],
                enhanced_stats['Max. Drawdown [%]']
            ]
        }).set_index('Metric')
