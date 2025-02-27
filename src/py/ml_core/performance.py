import pandas as pd
from backtesting import Backtest

class EnhancedPerformanceAnalyzer:
    def __init__(self, strategy):
        self.bt = Backtest(strategy)
        
    def analyze_enhancements(self, data):
        # Test base vs enhanced
        base_stats = self.bt.run(data.query("days_since_dividend > 30 and split_ratio == 1.0"))
        enhanced_stats = self.bt.run(data)
        
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
