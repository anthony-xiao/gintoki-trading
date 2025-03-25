# src/py/ml_core/ensemble_strategy.py
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Tuple
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
        """Generate signals with enhanced market context and model integration"""
        try:
            # Validate input data
            if processed_data is None or processed_data.empty:
                logger.warning("Empty or None data received in calculate_signals")
                return self._get_default_signals()
            
            # Ensure we have enough data for sequences
            if len(processed_data) < self.seq_length:
                logger.warning(f"Insufficient data for sequences. Need {self.seq_length}, got {len(processed_data)}")
                return self._get_default_signals()
            
            latest_data = processed_data.iloc[-self.seq_length:]
            
            # Get market regime
            regime_probs = np.array([0.33, 0.33, 0.34])  # Default to neutral regime
            if self.models['lstm_volatility'] is not None:
                try:
                    sequences = self._create_sequences(latest_data[self.feature_columns])
                    if len(sequences) > 0:
                        regime_probs = self.models['lstm_volatility'].predict(sequences, verbose=0)
                except Exception as e:
                    logger.warning(f"Error in LSTM volatility prediction: {str(e)}")
            
            current_regime = self._detect_regime(regime_probs[-1])
            
            # Get component signals with confidence scores
            component_signals = {}
            if self.models['xgb_momentum'] is not None:
                try:
                    xgb_signal, xgb_conf = self._xgb_momentum_signal(latest_data)
                    component_signals['xgb'] = {'signal': xgb_signal, 'confidence': xgb_conf}
                except Exception as e:
                    logger.warning(f"Error in XGB momentum prediction: {str(e)}")
                    component_signals['xgb'] = {'signal': 0.0, 'confidence': 0.0}
            
            if self.models['lstm_volatility'] is not None:
                try:
                    lstm_signal, lstm_conf = self._lstm_volatility_signal(latest_data)
                    component_signals['lstm'] = {'signal': lstm_signal, 'confidence': lstm_conf}
                except Exception as e:
                    logger.warning(f"Error in LSTM signal prediction: {str(e)}")
                    component_signals['lstm'] = {'signal': 0.0, 'confidence': 0.0}
            
            if self.models['transformer_trend'] is not None:
                try:
                    transformer_signal, transformer_conf = self._transformer_trend_signal(latest_data)
                    component_signals['transformer'] = {'signal': transformer_signal, 'confidence': transformer_conf}
                except Exception as e:
                    logger.warning(f"Error in transformer prediction: {str(e)}")
                    component_signals['transformer'] = {'signal': 0.0, 'confidence': 0.0}
            
            # Calculate market context factors
            market_context = self._calculate_market_context(latest_data)
            
            # Combine signals with regime-based weights and confidence
            weights = self.regime_weights[current_regime]
            combined = 0.0
            total_weight = 0.0
            
            for model, weight in weights.items():
                if model in component_signals:
                    signal = component_signals[model]['signal']
                    confidence = component_signals[model]['confidence']
                    # Apply confidence-based weighting
                    adjusted_weight = weight * confidence
                    combined += signal * adjusted_weight
                    total_weight += adjusted_weight
            
            # Normalize combined signal
            if total_weight > 0:
                combined /= total_weight
            
            # Apply market context adjustments
            combined *= market_context['trend_factor'] * market_context['liquidity_factor']
            
            # Apply risk adjustments
            try:
                risk_factor = self._calculate_risk_factor(latest_data)
                final_signal = combined * risk_factor
            except Exception as e:
                logger.warning(f"Error in risk calculation: {str(e)}")
                final_signal = combined
            
            # Anomaly detection
            if self.models['isolation_forest'] is not None:
                try:
                    if self._detect_anomalies(latest_data):
                        final_signal *= 0.5
                except Exception as e:
                    logger.warning(f"Error in anomaly detection: {str(e)}")

            # Calculate overall confidence
            overall_confidence = np.mean([comp['confidence'] for comp in component_signals.values()])
            
            return {
                'signal': np.tanh(final_signal),
                'regime': current_regime,
                'confidence': overall_confidence,
                'components': {
                    model: comp['signal'] 
                    for model, comp in component_signals.items()
                },
                'market_context': market_context
            }
            
        except Exception as e:
            logger.error(f"Error in calculate_signals: {str(e)}")
            return self._get_default_signals()

    def _get_default_signals(self) -> Dict:
        """Return default signals when errors occur"""
        return {
            'signal': 0.0,
            'regime': 'neutral',
            'confidence': 0.0,
            'components': {
                'transformer': 0.0,
                'xgb': 0.0,
                'lstm': 0.0
            },
            'market_context': {
                'trend_factor': 1.0,
                'liquidity_factor': 1.0
            }
        }

    def _calculate_market_context(self, data: pd.DataFrame) -> Dict:
        """Calculate market context factors"""
        try:
            # Calculate trend strength
            returns = data['returns'].iloc[-20:]
            trend_strength = abs(returns.mean() / (returns.std() + 1e-6))
            trend_factor = np.clip(1.0 / (1.0 + trend_strength), 0.7, 1.3)
            
            # Calculate liquidity factor
            volume_ma = data['volume'].rolling(20).mean().iloc[-1]
            volume_std = data['volume'].rolling(20).std().iloc[-1]
            liquidity_factor = np.clip(volume_ma / (volume_std + 1e-6), 0.5, 1.5)
            
            return {
                'trend_factor': trend_factor,
                'liquidity_factor': liquidity_factor
            }
        except Exception as e:
            logger.warning(f"Error calculating market context: {str(e)}")
            return {'trend_factor': 1.0, 'liquidity_factor': 1.0}

    def update_model_weights(self, performance_metrics: Dict[str, float], window_size: int = 20) -> None:
        """Update model weights based on recent performance metrics"""
        try:
            # Calculate performance scores for each model
            model_scores = {}
            for model, metrics in performance_metrics.items():
                if model in self.regime_weights['neutral']:  # Only update weights for models in ensemble
                    # Calculate composite score from metrics
                    score = (
                        0.4 * metrics.get('accuracy', 0.0) +
                        0.3 * metrics.get('sharpe_ratio', 0.0) +
                        0.3 * metrics.get('win_rate', 0.0)
                    )
                    model_scores[model] = max(0.0, score)  # Ensure non-negative
            
            # Normalize scores
            total_score = sum(model_scores.values())
            if total_score > 0:
                model_scores = {
                    model: score / total_score 
                    for model, score in model_scores.items()
                }
                
                # Update weights for each regime with smoothing
                for regime in self.regime_weights:
                    current_weights = self.regime_weights[regime]
                    for model in current_weights:
                        if model in model_scores:
                            # Smooth weight updates
                            current_weights[model] = (
                                0.7 * current_weights[model] +
                                0.3 * model_scores[model]
                            )
                
                # Normalize weights for each regime
                for regime in self.regime_weights:
                    total = sum(self.regime_weights[regime].values())
                    self.regime_weights[regime] = {
                        model: weight / total 
                        for model, weight in self.regime_weights[regime].items()
                    }
                
                logger.info(f"Updated model weights based on performance: {model_scores}")
                
        except Exception as e:
            logger.error(f"Error updating model weights: {str(e)}")

    def _transformer_trend_signal(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Get transformer-based trend strength"""
        sequences = self._create_sequences(data[self.feature_columns])
        return float(self.models['transformer_trend'].predict(sequences)[-1][0]), 1.0

    def _xgb_momentum_signal(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Existing XGBoost momentum signal (unchanged)"""
        features = data[self.feature_columns].values[-100:]
        return self.models['xgb_momentum'].predict_proba(features)[:,-1].mean(), 1.0

    def _lstm_volatility_signal(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Existing LSTM volatility signal (unchanged)"""
        sequences = self._create_sequences(data[self.feature_columns])
        return float(self.models['lstm_volatility'].predict(sequences)[-1][0]), 1.0

    def _detect_regime(self, probabilities: np.ndarray) -> str:
        """Existing regime detection (unchanged)"""
        if probabilities[0] > 0.7: return 'high_volatility'
        if probabilities[1] < 0.3: return 'low_volatility'
        return 'neutral'

    def _adjust_regime_weights(self, current_regime: str, recent_performance: Dict[str, float]) -> None:
        """Dynamically adjust regime weights based on recent performance"""
        # Calculate performance ratios
        total_performance = sum(recent_performance.values())
        if total_performance > 0:
            performance_ratios = {
                model: perf / total_performance 
                for model, perf in recent_performance.items()
            }
            
            # Update weights for current regime
            current_weights = self.regime_weights[current_regime]
            for model in current_weights:
                if model in performance_ratios:
                    # Smooth the weight update
                    current_weights[model] = 0.7 * current_weights[model] + 0.3 * performance_ratios[model]
            
            # Normalize weights
            total = sum(current_weights.values())
            self.regime_weights[current_regime] = {
                model: weight / total 
                for model, weight in current_weights.items()
            }
            
            logger.info(f"Updated weights for {current_regime}: {self.regime_weights[current_regime]}")

    def _create_sequences(self, data: pd.DataFrame) -> np.ndarray:
        """Create input sequences (unchanged)"""
        return np.array([data.iloc[i-self.seq_length:i].values 
                       for i in range(self.seq_length, len(data))])

    def _calculate_risk_factor(self, data: pd.DataFrame) -> float:
        """Enhanced risk calculation with multiple factors"""
        try:
            # Get base volatility and spread factors
            vol = data['volatility'].iloc[-1]
            spread = data['spread_ratio'].iloc[-1]
            
            # Calculate volume-based liquidity factor
            volume_ma = data['volume'].rolling(20).mean().iloc[-1]
            volume_std = data['volume'].rolling(20).std().iloc[-1]
            volume_factor = np.clip(volume_ma / (volume_std + 1e-6), 0.5, 1.5)
            
            # Calculate trend strength factor
            returns = data['returns'].iloc[-20:]
            trend_strength = abs(returns.mean() / (returns.std() + 1e-6))
            trend_factor = np.clip(1.0 / (1.0 + trend_strength), 0.7, 1.3)
            
            # Calculate market regime factor
            regime_factor = 1.0
            if self._detect_regime(self.models['lstm_volatility'].predict(
                self._create_sequences(data[self.feature_columns])[-1:], verbose=0
            )) == 'high_volatility':
                regime_factor = 0.8
            
            # Combine all factors with weights
            vol_factor = np.clip(self.risk_params['max_vol'] / vol, 0.5, 1.5)
            spread_factor = np.clip(self.risk_params['max_spread'] / spread, 0.7, 1.3)
            
            # Weighted combination of factors
            risk_factor = (
                0.3 * vol_factor +
                0.2 * spread_factor +
                0.2 * volume_factor +
                0.15 * trend_factor +
                0.15 * regime_factor
            )
            
            # Apply non-linear scaling for more conservative risk management
            risk_factor = np.tanh(risk_factor - 1) * 0.5 + 0.5
            
            return risk_factor
            
        except Exception as e:
            logger.warning(f"Error in risk calculation: {str(e)}")
            return 0.5  # Conservative default

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
