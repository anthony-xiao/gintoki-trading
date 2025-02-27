import numpy as np
import pandas as pd
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from .data_loader import EnhancedDataLoader

class EnhancedEnsembleTrader:
    def __init__(self):
        self.data_loader = EnhancedDataLoader()
        self.scaler = RobustScaler()
        self.models = {
            'lstm': tf.keras.models.load_model('regime_model.h5'),
            'xgb': XGBClassifier(n_estimators=500, learning_rate=0.01),
            'rf': RandomForestClassifier(n_estimators=300, max_depth=10)
        }
        self.feature_mask = np.load('feature_mask.npz')['mask']
        
    def _enhance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add corporate action and spread enhancements"""
        df['div_alert'] = (df['days_since_dividend'] < 5).astype(int)
        df['split_alert'] = (df['split_ratio'] != 1.0).astype(int)
        df['spread_ratio'] = df['bid_ask_spread'] / df['bid_ask_spread'].rolling(20).mean()
        return df

    def predict(self, X: pd.DataFrame) -> float:
        """Generate enhanced trading signal"""
        df = self._enhance_features(X.copy())
        features = self.scaler.transform(df.iloc[:, self.feature_mask])
        
        # Get regime context
        regime_probs = self.models['lstm'].predict(features.reshape(1, -1, len(self.feature_mask)))
        regime = np.argmax(regime_probs)
        
        # Generate base signal
        if regime == 0:  # Mean Reverting
            signal = self._mean_reversion_signal(df, features)
        elif regime == 2:  # Momentum
            signal = self._momentum_signal(df, features)
        else:
            signal = 0.0
            
        # Apply spread adjustment
        return signal * self._spread_adjustment_factor(df)

    def _mean_reversion_signal(self, df: pd.DataFrame, features: np.ndarray) -> float:
        """Mean reversion strategy with dividend boost"""
        xgb_prob = self.models['xgb'].predict_proba(features)[:, 1][0]
        rf_prob = self.models['rf'].predict_proba(features)[:, 1][0]
        base_signal = (xgb_prob + rf_prob) / 2
        
        # Dividend boost
        if df['div_alert'].iloc[-1]:
            return base_signal * 1.25
        return base_signal

    def _momentum_signal(self, df: pd.DataFrame, features: np.ndarray) -> float:
        """Momentum strategy with split boost"""
        xgb_prob = self.models['xgb'].predict_proba(features)[:, 1][0]
        rf_prob = self.models['rf'].predict_proba(features)[:, 1][0]
        base_signal = np.maximum(xgb_prob, rf_prob)
        
        # Split boost
        if df['split_alert'].iloc[-1]:
            return base_signal * 1.35
        return base_signal

    def _spread_adjustment_factor(self, df: pd.DataFrame) -> float:
        """Reduce position size in high spread environments"""
        spread_ratio = df['spread_ratio'].iloc[-1]
        return np.clip(1.2 - spread_ratio, 0.5, 1.0)

    def train(self, tickers: List[str]):
        """Train ensemble components"""
        data = pd.concat([self.data_loader.load_ticker_data(t) for t in tickers]).dropna()
        
        # Prepare features and labels
        X = self.scaler.fit_transform(data[self.data_loader.feature_columns])
        y = (data['close'].shift(-1) > data['close']).astype(int)
        
        # Train classifiers on selected features
        self.models['xgb'].fit(X[:, self.feature_mask], y)
        self.models['rf'].fit(X[:, self.feature_mask], y)
