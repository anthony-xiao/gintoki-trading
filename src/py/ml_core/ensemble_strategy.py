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
                    'transformer_optimized': load_model(config.get('transformer_optimized_model_path', config['transformer_model_path'])),
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
                'transformer_optimized': None,
                'isolation_forest': IsolationForest(contamination=0.05)
            }

    def set_models(self, models: Dict[str, tf.keras.Model]) -> None:
        """Set models after initialization"""
        for name, model in models.items():
            if name in self.models:
                self.models[name] = model
            else:
                raise ValueError(f"Unknown model name: {name}")

    def preprocess_data(self, raw_data: pd.DataFrame, feature_mask: np.ndarray = None, feature_metadata: Dict = None) -> pd.DataFrame:
        """Enhance data with derived features and optional feature masking"""
        processed = raw_data.copy()
        
        # Log initial data shape and columns
        logger.info(f"Initial data shape: {processed.shape}")
        logger.info(f"Initial columns: {processed.columns.tolist()}")
        
        # Check if we have enough data for calculations
        if len(processed) < 2:
            logger.warning("Insufficient data for feature calculations. Using actual values.")
            # For single row, use actual values
            processed['returns'] = 0.0  # Can't calculate returns without previous data
            processed['volatility'] = processed.get('volatility', 0.0)  # Use existing volatility if available
            processed['true_range'] = processed['high'] - processed['low']  # Use actual high-low range
            processed['volume_z'] = 1.0  # Use a neutral value instead of 0
            processed['spread_ratio'] = processed['bid_ask_spread'] / processed['mid_price']  # Use actual spread ratio
        else:
            # Use mid_price as close price if available, otherwise use close
            price_col = 'mid_price' if 'mid_price' in processed.columns else 'close'
            
            # Calculate returns with proper handling of edge cases
            processed['returns'] = processed[price_col].pct_change()
            processed['returns'] = processed['returns'].replace([np.inf, -np.inf], np.nan)
            processed['returns'] = processed['returns'].ffill().bfill()
            
            # Calculate volatility with proper window
            processed['volatility'] = processed['returns'].rolling(window=20, min_periods=1).std()
            processed['volatility'] = processed['volatility'].replace([np.inf, -np.inf], np.nan)
            processed['volatility'] = processed['volatility'].ffill().bfill()
            
            # Calculate true range with proper handling of edge cases
            high_low = processed['high'] - processed['low']
            high_close = np.abs(processed['high'] - processed[price_col].shift())
            low_close = np.abs(processed['low'] - processed[price_col].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            processed['true_range'] = np.max(ranges, axis=1)
            processed['true_range'] = processed['true_range'].ffill().bfill()
            
            # Calculate volume_z with improved normalization
            # First ensure volume is not zero or negative
            processed['volume'] = processed['volume'].replace(0, np.nan)
            processed['volume'] = processed['volume'].fillna(processed['volume'].mean())
            
            # Calculate rolling statistics with a minimum number of periods
            volume_ma = processed['volume'].rolling(window=50, min_periods=1).mean()
            volume_std = processed['volume'].rolling(window=50, min_periods=1).std()
            
            # Add a small constant to avoid division by zero
            epsilon = volume_ma * 1e-6
            processed['volume_z'] = (processed['volume'] - volume_ma) / (volume_std + epsilon)
            
            # Clip extreme values to prevent outliers
            processed['volume_z'] = processed['volume_z'].clip(-5, 5)
            
            # Forward fill any remaining NaN values
            processed['volume_z'] = processed['volume_z'].ffill().bfill()
            
            # Calculate spread ratio with proper handling of edge cases
            processed['spread_ratio'] = processed['bid_ask_spread'] / processed[price_col]
            processed['spread_ratio'] = processed['spread_ratio'].replace([np.inf, -np.inf], np.nan)
            processed['spread_ratio'] = processed['spread_ratio'].ffill().bfill()
        
        # Log shape after adding derived features
        logger.info(f"Data shape after adding derived features: {processed.shape}")
        
        # Log statistics of derived features
        logger.info("Derived feature statistics:")
        for feature in ['returns', 'volatility', 'true_range', 'volume_z', 'spread_ratio']:
            stats = processed[feature].describe()
            logger.info(f"  {feature}:")
            logger.info(f"    - Mean: {stats['mean']:.4f}")
            logger.info(f"    - Std: {stats['std']:.4f}")
            logger.info(f"    - Min: {stats['min']:.4f}")
            logger.info(f"    - Max: {stats['max']:.4f}")
            logger.info(f"    - NaN count: {processed[feature].isna().sum()}")
            # Log a sample of values to verify they're not all zeros
            logger.info(f"    - Sample values: {processed[feature].head().tolist()}")
        
        # Apply feature mask if provided
        if feature_mask is not None and feature_metadata is not None:
            selected_features = feature_metadata.get('selected_features', self.feature_columns)
            # Ensure all required derived features are included
            selected_features = list(set(selected_features + ['returns', 'volatility', 'true_range', 'volume_z', 'spread_ratio']))
            
            # Log feature selection
            logger.info(f"Selected features: {selected_features}")
            logger.info(f"Available columns: {processed.columns.tolist()}")
            
            # Verify all selected features exist
            missing_features = [f for f in selected_features if f not in processed.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            processed = processed[selected_features]
            
            # Log shape after feature selection
            logger.info(f"Data shape after feature selection: {processed.shape}")
        
        # Handle NaN values more carefully
        # First, identify which columns have NaN values and how many
        nan_counts = processed.isna().sum()
        nan_cols = nan_counts[nan_counts > 0].index.tolist()
        
        if nan_cols:
            logger.info("NaN value counts by column:")
            for col in nan_cols:
                count = nan_counts[col]
                percentage = (count / len(processed)) * 100
                logger.info(f"  {col}: {count} NaN values ({percentage:.2f}%)")
            
            # For derived features, we expect some NaN values at the start
            # For other features, we should investigate
            for col in nan_cols:
                if col not in ['returns', 'volatility', 'true_range', 'volume_z', 'spread_ratio']:
                    logger.warning(f"Unexpected NaN values in column: {col}")
            
            # Handle NaN values with a more robust approach
            for col in nan_cols:
                if col in ['returns', 'volatility', 'true_range', 'volume_z', 'spread_ratio']:
                    # For derived features, use forward fill then backward fill
                    processed[col] = processed[col].ffill().bfill()
                else:
                    # For other features, use median imputation
                    processed[col] = processed[col].fillna(processed[col].median())
            
            # Log shape after handling NaN values
            logger.info(f"Data shape after handling NaN values: {processed.shape}")
            
            # Verify no NaN values remain
            if processed.isna().any().any():
                remaining_nans = processed.isna().sum()
                logger.warning("Some NaN values remain after handling:")
                for col, count in remaining_nans.items():
                    if count > 0:
                        logger.warning(f"  {col}: {count} NaN values")
                        # Try one final time to handle any remaining NaNs
                        if col in ['returns', 'volatility', 'true_range', 'volume_z', 'spread_ratio']:
                            processed[col] = processed[col].ffill().bfill()
                        else:
                            processed[col] = processed[col].fillna(processed[col].median())
            
            # Final verification
            if processed.isna().any().any():
                raise ValueError("Unable to handle all NaN values")
        
        # Verify we still have data
        if len(processed) == 0:
            raise ValueError("No valid data remains after preprocessing")
        
        return processed

    def calculate_signals(self, processed_data: pd.DataFrame, volatility_model: tf.keras.Model = None, transformer_model: tf.keras.Model = None) -> Dict:
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
            
            # Create sequences for the entire dataset
            sequences = self._create_sequences(processed_data[self.feature_columns])
            if len(sequences) == 0:
                logger.warning("No valid sequences created")
                return self._get_default_signals()
            
            # Get market regime for each sequence
            regime_probs = np.array([0.33, 0.33, 0.34])  # Default to neutral regime
            if volatility_model is not None:
                try:
                    # Process sequences in batches to avoid memory issues
                    batch_size = 128
                    regime_probs_list = []
                    for i in range(0, len(sequences), batch_size):
                        batch = sequences[i:i+batch_size]
                        batch_probs = volatility_model.predict(batch, verbose=0)
                        regime_probs_list.append(batch_probs)
                    regime_probs = np.concatenate(regime_probs_list)
                except Exception as e:
                    logger.warning(f"Error in volatility prediction: {str(e)}")
                    regime_probs = np.tile([0.33, 0.33, 0.34], (len(sequences), 1))
            
            # Get component signals with confidence scores for each sequence
            component_signals = {}
            
            # XGB signals
            if self.models['xgb_momentum'] is not None:
                try:
                    features = processed_data[self.feature_columns].values
                    xgb_probs = self.models['xgb_momentum'].predict_proba(features)
                    xgb_signals = xgb_probs[:, -1]
                    component_signals['xgb'] = {
                        'signal': xgb_signals,
                        'confidence': np.ones_like(xgb_signals)
                    }
                except Exception as e:
                    logger.warning(f"Error in XGB momentum prediction: {str(e)}")
                    component_signals['xgb'] = {'signal': np.zeros(len(sequences)), 'confidence': np.zeros(len(sequences))}
            
            # LSTM signals
            if volatility_model is not None:
                try:
                    # Process sequences in batches
                    batch_size = 32
                    lstm_signals_list = []
                    for i in range(0, len(sequences), batch_size):
                        batch = sequences[i:i+batch_size]
                        batch_signals = volatility_model.predict(batch, verbose=0)
                        lstm_signals_list.append(batch_signals)
                    lstm_signals = np.concatenate(lstm_signals_list)
                    component_signals['lstm'] = {
                        'signal': lstm_signals.flatten(),
                        'confidence': np.ones_like(lstm_signals.flatten())
                    }
                except Exception as e:
                    logger.warning(f"Error in LSTM signal prediction: {str(e)}")
                    component_signals['lstm'] = {'signal': np.zeros(len(sequences)), 'confidence': np.zeros(len(sequences))}
            
            # Transformer signals
            if transformer_model is not None:
                try:
                    # Process sequences in batches
                    batch_size = 32
                    transformer_signals_list = []
                    for i in range(0, len(sequences), batch_size):
                        batch = sequences[i:i+batch_size]
                        batch_signals = transformer_model.predict(batch, verbose=0)
                        transformer_signals_list.append(batch_signals)
                    transformer_signals = np.concatenate(transformer_signals_list)
                    component_signals['transformer'] = {
                        'signal': transformer_signals.flatten(),
                        'confidence': np.ones_like(transformer_signals.flatten())
                    }
                except Exception as e:
                    logger.warning(f"Error in transformer prediction: {str(e)}")
                    component_signals['transformer'] = {'signal': np.zeros(len(sequences)), 'confidence': np.zeros(len(sequences))}
            
            # Calculate market context factors for each sequence
            market_contexts = []
            for i in range(len(sequences)):
                data_window = processed_data.iloc[i:i+self.seq_length]
                market_contexts.append(self._calculate_market_context(data_window))
            
            # Combine signals with regime-based weights and confidence
            combined_signals = np.zeros(len(sequences))
            total_weights = np.zeros(len(sequences))
            
            for i in range(len(sequences)):
                current_regime = self._detect_regime(regime_probs[i])
                weights = self.regime_weights[current_regime]
                
                for model, weight in weights.items():
                    if model in component_signals:
                        signal = component_signals[model]['signal'][i]
                        confidence = component_signals[model]['confidence'][i]
                        adjusted_weight = weight * confidence
                        combined_signals[i] += signal * adjusted_weight
                        total_weights[i] += adjusted_weight
            
            # Normalize combined signals
            valid_weights = total_weights > 0
            combined_signals[valid_weights] /= total_weights[valid_weights]
            
            # Apply market context adjustments
            for i in range(len(sequences)):
                combined_signals[i] *= market_contexts[i]['trend_factor'] * market_contexts[i]['liquidity_factor']
            
            # Apply risk adjustments
            risk_factors = np.array([self._calculate_risk_factor(processed_data.iloc[i:i+self.seq_length]) 
                                   for i in range(len(sequences))])
            combined_signals *= risk_factors
            
            # Anomaly detection
            if self.models['isolation_forest'] is not None:
                try:
                    # Ensure anomaly features match the sequence length
                    anomaly_features = processed_data[['returns', 'volume_z', 'spread_ratio']].values[:len(sequences)]
                    anomalies = self.models['isolation_forest'].predict(anomaly_features) == -1
                    combined_signals[anomalies] *= 0.5
                except Exception as e:
                    logger.warning(f"Error in anomaly detection: {str(e)}")

            # Calculate overall confidence
            overall_confidence = np.mean([comp['confidence'] for comp in component_signals.values()], axis=0)
            
            # Ensure all outputs are arrays
            return {
                'signal': np.array(combined_signals),
                'regime': np.array([self._detect_regime(probs) for probs in regime_probs]),
                'confidence': np.array(overall_confidence),
                'components': {
                    model: np.array(comp['signal']) 
                    for model, comp in component_signals.items()
                },
                'market_context': market_contexts
            }
            
        except Exception as e:
            logger.error(f"Error in calculate_signals: {str(e)}")
            # Return default signals as arrays
            return {
                'signal': np.array([0.0]),
                'regime': np.array(['neutral']),
                'confidence': np.array([0.0]),
                'components': {
                    'transformer': np.array([0.0]),
                    'xgb': np.array([0.0]),
                    'lstm': np.array([0.0])
                },
                'market_context': [{
                    'trend_factor': 1.0,
                    'liquidity_factor': 1.0
                }]
            }

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

    def _transformer_trend_signal(self, data: pd.DataFrame, transformer_model: tf.keras.Model = None) -> Tuple[float, float]:
        """Get transformer-based trend strength"""
        sequences = self._create_sequences(data[self.feature_columns])
        if transformer_model is not None:
            return float(transformer_model.predict(sequences)[-1][0]), 1.0
        elif self.models['transformer_trend'] is not None:
            return float(self.models['transformer_trend'].predict(sequences)[-1][0]), 1.0
        return 0.0, 0.0

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
            liquidity_factor = np.clip(volume_ma / (volume_std + 1e-6), 0.5, 1.5)
            
            # Calculate trend strength factor
            returns = data['returns'].iloc[-20:]
            trend_strength = abs(returns.mean() / (returns.std() + 1e-6))
            trend_factor = np.clip(1.0 / (1.0 + trend_strength), 0.7, 1.3)
            
            # Calculate market regime factor - default to neutral if model not available
            regime_factor = 1.0
            if self.models['lstm_volatility'] is not None:
                try:
                    sequences = self._create_sequences(data[self.feature_columns])
                    if len(sequences) > 0:
                        regime_probs = self.models['lstm_volatility'].predict(sequences[-1:], verbose=0)
                        if self._detect_regime(regime_probs[0]) == 'high_volatility':
                            regime_factor = 0.8
                except Exception as e:
                    logger.warning(f"Error in regime detection for risk calculation: {str(e)}")
            
            # Combine all factors with weights
            vol_factor = np.clip(self.risk_params['max_vol'] / vol, 0.5, 1.5)
            spread_factor = np.clip(self.risk_params['max_spread'] / spread, 0.7, 1.3)
            
            # Weighted combination of factors
            risk_factor = (
                0.3 * vol_factor +
                0.2 * spread_factor +
                0.2 * liquidity_factor +
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
