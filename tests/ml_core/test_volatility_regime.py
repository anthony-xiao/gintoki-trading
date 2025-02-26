import unittest
import numpy as np
from ml_core.volatility_regime import VolatilityRegimeDetector

class TestVolatilityDetection(unittest.TestCase):
    def setUp(self):
        self.model = VolatilityRegimeDetector(lookback=10)
        self.sample_data = np.random.randn(1000, 4)  # Mock features
    
    def test_regime_labeling(self):
        df = pd.DataFrame({'close': np.exp(np.cumsum(np.random.normal(0, 0.01, 1000)))})
        labeled = self.model.label_regimes(df)
        self.assertIn('regime', labeled.columns)
        self.assertTrue(labeled['regime'].isin([0,1,2]).all())
    
    def test_sequence_creation(self):
        X, y = self.model.create_sequences(pd.DataFrame(self.sample_data))
        self.assertEqual(X.shape[0], 990)
        self.assertEqual(X.shape[2], 4)

if __name__ == '__main__':
    unittest.main()
