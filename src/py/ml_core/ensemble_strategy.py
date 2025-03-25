# src/py/ml_core/ensemble_strategy.py
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import load_model
import logging

logger = logging.getLogger(__name__)

class AdaptiveEnsembleTrader:
    def __init__(self, config: Dict, skip_model_loading: bool = False):
        """Initialize the ensemble trader with optional model loading"""
        self.feature_columns = config['feature_columns']
        self.regime_weights = config.get('regime_weights', {
            'high_volatility': {'transformer': 0.6, 'xgb': 0.3, 'lstm': 0.1},
            'low_volatility': {'transformer': 0.3, 'xgb': 0.5, 'lstm': 0.2},
            'neutral': {'transformer': 0.4, 'xgb': 0.4, 'lstm': 0.2}
        })
        self.risk_params = config['risk_management']
        self.seq_length = 60  # Matches transformer window size
        
        # Initialize models dictionary
        self.models = {}
        
        # Load models if not skipping
        if not skip_model_loading:
            if 'volatility_model_path' in config and 'transformer_model_path' in config:
                self.models = {
                    'lstm_volatility': load_model(config['volatility_model_path']),
                    'xgb_momentum': XGBClassifier(),
                    'transformer_trend': load_model(config['transformer_model_path']),
                    'isolation_forest': IsolationForest(contamination=0.05)
                }
            else:
                raise ValueError("Model paths required when not skipping model loading")
        else:
            # Initialize with empty models
            self.models = {
                'lstm_volatility': None,
                'xgb_momentum': XGBClassifier(),
                'transformer_trend': None,
                'isolation_forest': IsolationForest(contamination=0.05)
            }

    def set_models(self, models: Dict[str, tf.keras.Model]) -> None:
        """Set models after initialization"""
        for name, model in models.items():
            if name in self.models:
                self.models[name] = model
            else:
                raise ValueError(f"Unknown model name: {name}")

    def preprocess_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Enhance data with derived features (unchanged from original)"""
        processed = raw_data.copy()
        processed['returns'] = np.log(processed['close']/processed['close'].shift(1))
        processed['volatility'] = processed['returns'].rolling(20).std()
        processed['true_range'] = self._true_range(processed)
        processed['volume_z'] = (processed['volume'] - processed['volume'].rolling(50).mean()) \
                               / processed['volume'].rolling(50).std()
        processed['spread_ratio'] = processed['bid_ask_spread'] / processed['mid_price']
        return processed.dropna()

    def calculate_signals(self, processed_data: pd.DataFrame) -> Dict:
        """Generate signals with transformer integration"""
        try:
            # Validate input data
            if processed_data is None or processed_data.empty:
                logger.warning("Empty or None data received in calculate_signals")
                return {
                    'signal': 0.0,
                    'regime': 'neutral',
                    'confidence': 0.0,
                    'components': {
                        'transformer': 0.0,
                        'xgb': 0.0,
                        'lstm': 0.0
                    }
                }
            
            # Ensure we have enough data for sequences
            if len(processed_data) < self.seq_length:
                logger.warning(f"Insufficient data for sequences. Need {self.seq_length}, got {len(processed_data)}")
                return {
                    'signal': 0.0,
                    'regime': 'neutral',
                    'confidence': 0.0,
                    'components': {
                        'transformer': 0.0,
                        'xgb': 0.0,
                        'lstm': 0.0
                    }
                }
            
            latest_data = processed_data.iloc[-self.seq_length:]
            
            # Initialize default signals
            regime_probs = np.array([0.33, 0.33, 0.34])  # Default to neutral regime
            xgb_signal = 0.0
            lstm_signal = 0.0
            transformer_signal = 0.0
            
            # Get market regime if model exists and is trained
            if self.models['lstm_volatility'] is not None:
                try:
                    sequences = self._create_sequences(latest_data[self.feature_columns])
                    if len(sequences) > 0:
                        regime_probs = self.models['lstm_volatility'].predict(sequences, verbose=0)
                except Exception as e:
                    logger.warning(f"Error in LSTM volatility prediction: {str(e)}")
            
            current_regime = self._detect_regime(regime_probs[-1])
            
            # Get component signals if models exist and are trained
            if self.models['xgb_momentum'] is not None:
                try:
                    xgb_signal = self._xgb_momentum_signal(latest_data)
                except Exception as e:
                    logger.warning(f"Error in XGB momentum prediction: {str(e)}")
            
            if self.models['lstm_volatility'] is not None:
                try:
                    lstm_signal = self._lstm_volatility_signal(latest_data)
                except Exception as e:
                    logger.warning(f"Error in LSTM signal prediction: {str(e)}")
            
            if self.models['transformer_trend'] is not None:
                try:
                    transformer_signal = self._transformer_trend_signal(latest_data)
                except Exception as e:
                    logger.warning(f"Error in transformer prediction: {str(e)}")
            
            # Combine signals with regime-based weights
            weights = self.regime_weights[current_regime]
            combined = (
                weights['transformer'] * transformer_signal +
                weights['xgb'] * xgb_signal +
                weights['lstm'] * lstm_signal
            )
            
            # Apply risk adjustments
            try:
                risk_factor = self._calculate_risk_factor(latest_data)
                final_signal = combined * risk_factor
            except Exception as e:
                logger.warning(f"Error in risk calculation: {str(e)}")
                final_signal = combined
            
            # Anomaly detection if model exists
            if self.models['isolation_forest'] is not None:
                try:
                    if self._detect_anomalies(latest_data):
                        final_signal *= 0.5
                except Exception as e:
                    logger.warning(f"Error in anomaly detection: {str(e)}")

            return {
                'signal': np.tanh(final_signal),
                'regime': current_regime,
                'confidence': abs(final_signal),
                'components': {
                    'transformer': transformer_signal,
                    'xgb': xgb_signal,
                    'lstm': lstm_signal
                }
            }
            
        except Exception as e:
            logger.error(f"Error in calculate_signals: {str(e)}")
            return {
                'signal': 0.0,
                'regime': 'neutral',
                'confidence': 0.0,
                'components': {
                    'transformer': 0.0,
                    'xgb': 0.0,
                    'lstm': 0.0
                }
            }

    def _transformer_trend_signal(self, data: pd.DataFrame) -> float:
        """Get transformer-based trend strength"""
        sequences = self._create_sequences(data[self.feature_columns])
        return float(self.models['transformer_trend'].predict(sequences)[-1][0])

    def _xgb_momentum_signal(self, data: pd.DataFrame) -> float:
        """Existing XGBoost momentum signal (unchanged)"""
        features = data[self.feature_columns].values[-100:]
        return self.models['xgb_momentum'].predict_proba(features)[:,-1].mean()

    def _lstm_volatility_signal(self, data: pd.DataFrame) -> float:
        """Existing LSTM volatility signal (unchanged)"""
        sequences = self._create_sequences(data[self.feature_columns])
        return float(self.models['lstm_volatility'].predict(sequences)[-1][0])

    def _detect_regime(self, probabilities: np.ndarray) -> str:
        """Existing regime detection (unchanged)"""
        if probabilities[0] > 0.7: return 'high_volatility'
        if probabilities[1] < 0.3: return 'low_volatility'
        return 'neutral'

    def _create_sequences(self, data: pd.DataFrame) -> np.ndarray:
        """Create input sequences (unchanged)"""
        return np.array([data.iloc[i-self.seq_length:i].values 
                       for i in range(self.seq_length, len(data))])

    def _calculate_risk_factor(self, data: pd.DataFrame) -> float:
        """Existing risk calculation (unchanged)"""
        vol = data['volatility'].iloc[-1]
        spread = data['spread_ratio'].iloc[-1]
        vol_factor = np.clip(self.risk_params['max_vol'] / vol, 0.5, 1.5)
        spread_factor = np.clip(self.risk_params['max_spread'] / spread, 0.7, 1.3)
        return vol_factor * spread_factor

    def _detect_anomalies(self, data: pd.DataFrame) -> bool:
        """Existing anomaly detection (unchanged)"""
        features = data[['returns', 'volume_z', 'spread_ratio']].values
        return self.models['isolation_forest'].predict(features)[-1] == -1

    def _true_range(self, df: pd.DataFrame) -> pd.Series:
        """Existing TR calculation (unchanged)"""
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift()).abs()
        lc = (df['low'] - df['close'].shift()).abs()
        return pd.concat([hl, hc, lc], axis=1).max(axis=1)
