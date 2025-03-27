import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
import os
import boto3
from io import BytesIO
import json
from .data_loader import EnhancedDataLoader
from .volatility_regime import EnhancedVolatilityDetector
from .transformer_trend import TransformerTrendAnalyzer
from .ensemble_strategy import AdaptiveEnsembleTrader
import time
import tempfile

logger = logging.getLogger(__name__)

class CastLayer(tf.keras.layers.Layer):
    """Custom layer for type casting"""
    def __init__(self, dtype=tf.float32, **kwargs):
        super(CastLayer, self).__init__(**kwargs)
        self._dtype = dtype

    def call(self, inputs):
        return tf.cast(inputs, self._dtype)

    def get_config(self):
        config = super(CastLayer, self).get_config()
        config.update({'dtype': tf.keras.backend.standardize_dtype(self._dtype)})
        return config

    @classmethod
    def from_config(cls, config):
        dtype = config.pop('dtype', tf.float32)
        return cls(dtype=dtype, **config)

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

class ModelFactory:
    def __init__(self, config: Dict):
        """Initialize the model factory with configuration"""
        self.config = config
        self.data_loader = EnhancedDataLoader()
        self.models = {}
        self.ensemble = None
        self.s3_client = boto3.client('s3')
        self.bucket = config.get('s3_bucket', 'quant-trader-data-gintoki')
        self.model_prefix = config.get('model_prefix', 'models/')
        
    def _get_s3_key(self, model_name: str, version: Optional[str] = None, extension: str = '.h5') -> str:
        """Generate S3 key for model storage using timestamp"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        if version:
            return f"{self.model_prefix}{model_name}/models/{version}_{timestamp}{extension}"
        return f"{self.model_prefix}{model_name}/models/{timestamp}{extension}"
        
    def save_model_to_s3(self, model: tf.keras.Model, model_name: str, version: Optional[str] = None) -> str:
        """Save model to S3"""
        try:
            # Generate S3 key with timestamp
            s3_key = self._get_s3_key(model_name, version)
            
            # Create a temporary file with .h5 extension
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
                # Save model directly to temporary file
                model.save(temp_file.name)
                
                # Upload the temporary file to S3
                self.s3_client.upload_file(
                    temp_file.name,
                    self.bucket,
                    s3_key,
                    ExtraArgs={'ContentType': 'application/octet-stream'}
                )
                
                # Clean up temporary file
                os.unlink(temp_file.name)
            
            logger.info(f"Saved {model_name} model to s3://{self.bucket}/{s3_key}")
            return s3_key
            
        except Exception as e:
            logger.error(f"Failed to save model to S3: {str(e)}")
            raise
            
    def load_model_from_s3(self, model_name: str, version: Optional[str] = None) -> tf.keras.Model:
        """Load latest model from S3 based on timestamp"""
        try:
            # List all models in the directory
            prefix = f"{self.model_prefix}{model_name}/models/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                raise FileNotFoundError(f"No models found for {model_name}")
            
            # Get all model files
            model_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.h5')]
            
            if not model_files:
                raise FileNotFoundError(f"No .h5 models found for {model_name}")
            
            # If version specified, find the latest model with that version
            if version:
                version_files = [f for f in model_files if f.startswith(f"{prefix}{version}_")]
                if not version_files:
                    raise FileNotFoundError(f"No models found for version {version}")
                latest_model = max(version_files)  # Latest by timestamp
            else:
                latest_model = max(model_files)  # Latest by timestamp
            
            # Download model to memory
            model_buffer = BytesIO()
            self.s3_client.download_fileobj(self.bucket, latest_model, model_buffer)
            model_buffer.seek(0)
            
            # Create a temporary file with .h5 extension
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
                # Write the model data to the temporary file
                temp_file.write(model_buffer.getvalue())
                temp_file.flush()  # Ensure all data is written
                temp_path = temp_file.name
            
            try:
                # Define custom objects for model loading
                custom_objects = {
                    'Cast': CastLayer,  # Use our custom CastLayer
                    'CastLayer': CastLayer,  # Also register the class name
                    'Dense': tf.keras.layers.Dense,
                    'Dropout': tf.keras.layers.Dropout,
                    'LayerNormalization': tf.keras.layers.LayerNormalization,
                    'MultiHeadAttention': tf.keras.layers.MultiHeadAttention,
                    'HuberLoss': tf.keras.losses.Huber,
                    'Adam': tf.keras.optimizers.Adam,
                    'GRU': tf.keras.layers.GRU,
                    'LSTM': tf.keras.layers.LSTM,
                    'Bidirectional': tf.keras.layers.Bidirectional,
                    'Conv1D': tf.keras.layers.Conv1D,
                    'MaxPooling1D': tf.keras.layers.MaxPooling1D,
                    'GlobalAveragePooling1D': tf.keras.layers.GlobalAveragePooling1D,
                    'BatchNormalization': tf.keras.layers.BatchNormalization,
                    'Activation': tf.keras.layers.Activation,
                    'Add': tf.keras.layers.Add,
                    'Concatenate': tf.keras.layers.Concatenate,
                    'Flatten': tf.keras.layers.Flatten,
                    'Reshape': tf.keras.layers.Reshape,
                    'Embedding': tf.keras.layers.Embedding,
                    'TimeDistributed': tf.keras.layers.TimeDistributed,
                    'Lambda': tf.keras.layers.Lambda
                }
                
                # Load model from temporary file with custom objects
                model = tf.keras.models.load_model(temp_path, custom_objects=custom_objects)
                logger.info(f"Loaded {model_name} model from s3://{self.bucket}/{latest_model}")
                return model
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Error cleaning up temporary file: {str(e)}")
            
        except Exception as e:
            logger.error(f"Failed to load model from S3: {str(e)}")
            raise
            
    def save_ensemble_config_to_s3(self, config: Dict, version: Optional[str] = None) -> str:
        """Save ensemble configuration to S3"""
        try:
            config_buffer = BytesIO()
            config_buffer.write(json.dumps(config).encode())
            config_buffer.seek(0)
            
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            if version:
                s3_key = f"{self.model_prefix}ensemble/models/{version}_{timestamp}.json"
            else:
                s3_key = f"{self.model_prefix}ensemble/models/{timestamp}.json"
            
            self.s3_client.upload_fileobj(
                config_buffer,
                self.bucket,
                s3_key,
                ExtraArgs={'ContentType': 'application/json'}
            )
            
            logger.info(f"Saved ensemble config to s3://{self.bucket}/{s3_key}")
            return s3_key
            
        except Exception as e:
            logger.error(f"Failed to save ensemble config to S3: {str(e)}")
            raise
            
    def load_ensemble_config_from_s3(self, version: Optional[str] = None) -> Dict:
        """Load latest ensemble configuration from S3"""
        try:
            # List all configs in the directory
            prefix = f"{self.model_prefix}ensemble/models/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                raise FileNotFoundError("No ensemble configs found")
            
            # Get all config files
            config_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.json')]
            
            if not config_files:
                raise FileNotFoundError("No .json configs found")
            
            # If version specified, find the latest config with that version
            if version:
                version_files = [f for f in config_files if f.startswith(f"{prefix}{version}_")]
                if not version_files:
                    raise FileNotFoundError(f"No configs found for version {version}")
                latest_config = max(version_files)  # Latest by timestamp
            else:
                latest_config = max(config_files)  # Latest by timestamp
            
            # Download and load the latest config
            config_buffer = BytesIO()
            self.s3_client.download_fileobj(self.bucket, latest_config, config_buffer)
            config_buffer.seek(0)
            
            config = json.loads(config_buffer.getvalue().decode())
            logger.info(f"Loaded ensemble config from s3://{self.bucket}/{latest_config}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load ensemble config from S3: {str(e)}")
            raise
            
    def create_models(self) -> Dict:
        """Create and initialize all required models"""
        try:
            # Get sequence length from config
            seq_length = self.config.get('seq_length', 30)  # Default to 30 if not specified
            
            # 1. Create Volatility Detector
            logger.info(f"Creating Volatility Detector with sequence length {seq_length}...")
            self.models['volatility'] = EnhancedVolatilityDetector(
                lookback=seq_length
            )
            
            # 2. Create Transformer Trend Analyzer
            logger.info(f"Creating Transformer Trend Analyzer with sequence length {seq_length}...")
            self.models['transformer'] = TransformerTrendAnalyzer(
                seq_length=seq_length,
                d_model=self.config.get('d_model', 64),
                num_heads=self.config.get('num_heads', 8)
            )
            
            # 3. Create Optimized Transformer
            logger.info(f"Creating Optimized Transformer with sequence length {seq_length}...")
            self.models['transformer_optimized'] = TransformerTrendAnalyzer(
                seq_length=seq_length,
                d_model=self.config.get('d_model', 64),
                num_heads=self.config.get('num_heads', 8)
            )
            
            # 4. Create Adaptive Ensemble with minimal config
            logger.info("Creating Adaptive Ensemble...")
            ensemble_config = {
                'feature_columns': self.data_loader.feature_columns,
                'risk_management': self.config.get('risk_management', {
                    'max_vol': 0.015,
                    'max_spread': 0.002
                }),
                'regime_weights': self.config.get('regime_weights', {
                    'high_volatility': {'transformer': 0.6, 'xgb': 0.3, 'lstm': 0.1},
                    'low_volatility': {'transformer': 0.3, 'xgb': 0.5, 'lstm': 0.2},
                    'neutral': {'transformer': 0.4, 'xgb': 0.4, 'lstm': 0.2}
                })
            }
            
            # Initialize ensemble without models
            self.ensemble = AdaptiveEnsembleTrader(ensemble_config, skip_model_loading=True)
            
            # Set the models after creation
            if hasattr(self.ensemble, 'set_models'):
                self.ensemble.set_models({
                    'lstm_volatility': self.models['volatility'].model,
                    'transformer_trend': self.models['transformer'].model,
                    'transformer_optimized': self.models['transformer_optimized'].model
                })
            
            logger.info("✅ All models created successfully")
            return self.models
            
        except Exception as e:
            logger.error(f"Failed to create models: {str(e)}")
            raise
            
    def save_models(self, version: Optional[str] = None) -> Dict[str, str]:
        """Save all models to S3"""
        try:
            model_paths = {}
            
            # Save each model
            for model_name, model in self.models.items():
                if hasattr(model, 'model'):
                    s3_key = self.save_model_to_s3(model.model, model_name, version)
                    model_paths[model_name] = s3_key
            
            # Save ensemble configuration
            if self.ensemble:
                ensemble_config = {
                    'feature_columns': self.data_loader.feature_columns,
                    'volatility_model_path': model_paths.get('volatility'),
                    'transformer_model_path': model_paths.get('transformer'),
                    'transformer_optimized_model_path': model_paths.get('transformer_optimized'),
                    'risk_management': self.config.get('risk_management'),
                    'regime_weights': self.config.get('regime_weights')
                }
                s3_key = self.save_ensemble_config_to_s3(ensemble_config, version)
                model_paths['ensemble'] = s3_key
            
            return model_paths
            
        except Exception as e:
            logger.error(f"Failed to save models to S3: {str(e)}")
            raise
            
    def load_models(self, version: Optional[str] = None) -> None:
        """Load pre-trained models from S3"""
        try:
            # Load individual models
            for model_name in ['volatility', 'transformer', 'transformer_optimized']:
                if model_name in self.models:
                    self.models[model_name].model = self.load_model_from_s3(model_name, version)
            
            # Try to load ensemble configuration, but don't fail if it doesn't exist
            try:
                ensemble_config = self.load_ensemble_config_from_s3(version)
            except FileNotFoundError:
                logger.warning("No ensemble config found, using default configuration")
                # Create default ensemble config
                ensemble_config = {
                    'feature_columns': self.data_loader.feature_columns,
                    'risk_management': self.config.get('risk_management', {
                        'max_vol': 0.015,
                        'max_spread': 0.002
                    }),
                    'regime_weights': self.config.get('regime_weights', {
                        'high_volatility': {'transformer': 0.6, 'xgb': 0.3, 'lstm': 0.1},
                        'low_volatility': {'transformer': 0.3, 'xgb': 0.5, 'lstm': 0.2},
                        'neutral': {'transformer': 0.4, 'xgb': 0.4, 'lstm': 0.2}
                    })
                }
            
            # Create new ensemble instance with the config
            self.ensemble = AdaptiveEnsembleTrader(ensemble_config, skip_model_loading=True)
            
            # Set the models in the ensemble
            if hasattr(self.ensemble, 'set_models'):
                self.ensemble.set_models({
                    'lstm_volatility': self.models['volatility'].model,
                    'transformer_trend': self.models['transformer'].model,
                    'transformer_optimized': self.models['transformer_optimized'].model
                })
            
            logger.info("✅ All models loaded successfully from S3")
            
        except Exception as e:
            logger.error(f"Failed to load models from S3: {str(e)}")
            raise
            
    def get_trading_signal(self, data: pd.DataFrame) -> Tuple[float, Dict]:
        """Generate trading signal using the ensemble"""
        try:
            if self.ensemble is None:
                raise ValueError("Ensemble not initialized")
                
            # Preprocess data
            processed_data = self.ensemble.preprocess_data(data)
            
            # Get ensemble signals
            signals = self.ensemble.calculate_signals(processed_data)
            
            # Extract final signal and metadata
            final_signal = signals['signal']
            metadata = {
                'regime': signals['regime'],
                'confidence': signals['confidence'],
                'components': signals['components']
            }
            
            return final_signal, metadata
            
        except Exception as e:
            logger.error(f"Failed to generate trading signal: {str(e)}")
            raise
            
    def update_ensemble_weights(self, performance_metrics: Dict[str, float]) -> None:
        """Update ensemble weights based on model performance"""
        try:
            if self.ensemble is None:
                raise ValueError("Ensemble not initialized")
                
            # Calculate new weights based on performance
            total_performance = sum(performance_metrics.values())
            if total_performance > 0:
                new_weights = {
                    model: perf / total_performance 
                    for model, perf in performance_metrics.items()
                }
                
                # Update regime weights
                for regime in self.ensemble.regime_weights:
                    self.ensemble.regime_weights[regime].update(new_weights)
                    
                logger.info(f"Updated ensemble weights: {new_weights}")
                
        except Exception as e:
            logger.error(f"Failed to update ensemble weights: {str(e)}")
            raise 