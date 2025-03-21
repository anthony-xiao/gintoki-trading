# src/py/ml_core/shap_optimizer.py
import shap
import tensorflow as tf
import numpy as np
import joblib
from tqdm import tqdm
from .data_loader import EnhancedDataLoader
from .model_registry import EnhancedModelRegistry
import pandas as pd
import boto3
from io import BytesIO
import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class EnhancedSHAPOptimizer:
    def __init__(self, model_path: Optional[str] = None, background_samples: int = 1000, 
                 background_data: Optional[pd.DataFrame] = None, ticker: str = 'SMCI',
                 data_loader: Optional[EnhancedDataLoader] = None):
        """Initialize SHAP optimizer for day trading"""
        self.registry = EnhancedModelRegistry()
        self.data_loader = data_loader if data_loader is not None else EnhancedDataLoader()
        self.ticker = ticker  # Store ticker
        
        # Day trading specific features
        self.feature_columns = self.data_loader.feature_columns
        self.essential_features = [
            'bid_ask_spread',  # Critical for execution
            'volume',          # Liquidity indicator
            'vwap',           # Price impact
            'days_since_dividend',  # Corporate actions
            'split_ratio'     # Corporate actions
        ]
        
        # Trading-specific parameters
        self.min_liquidity_threshold = 0.001  # Minimum bid-ask spread
        self.max_impact_threshold = 0.02      # Maximum price impact
        self.volume_weight = 1.5              # Weight for volume features
        
        # Load models
        if model_path is None:
            logger.info("No model path provided, fetching latest regime model from S3")
            model_path = self._get_latest_regime_model()
            logger.info(f"Using latest regime model: {model_path}")
        
        self.regime_model = self._load_model_from_s3(model_path)
        self.transformer_model = self._load_transformer_model()
        
        # Initialize SHAP explainers for each component
        background_data = self._prepare_background(background_data, background_samples)
        
        # Create masker for feature permutation
        # Reshape background data to match expected shape (samples, timesteps, features)
        n_samples, n_timesteps, n_features = background_data.shape
        masker = shap.maskers.Independent(data=background_data.reshape(n_samples, n_timesteps * n_features))
        
        # Calculate required max_evals based on feature count
        required_evals = 2 * (n_timesteps * n_features) + 1
        logger.info(f"Setting max_evals to {required_evals} for {n_timesteps * n_features} total features")
        
        self.regime_explainer = shap.PermutationExplainer(
            model=self._predict_regime,
            data=background_data.reshape(n_samples, n_timesteps * n_features),
            masker=masker,
            max_evals=required_evals
        )
        
        self.trend_explainer = shap.PermutationExplainer(
            model=self._predict_trend,
            data=background_data.reshape(n_samples, n_timesteps * n_features),
            masker=masker,
            max_evals=required_evals
        )
        
        logger.info("SHAP optimizer initialized with trading-specific features")
        logger.info(f"Essential features: {self.essential_features}")

    def _predict_regime(self, x: np.ndarray) -> np.ndarray:
        """Predict market regime with proper shape handling"""
        try:
            # Get the number of features
            n_features = len(self.feature_columns)
            
            # Calculate the number of timesteps
            n_timesteps = 60  # Model expects 60 timesteps
            
            # Reshape input to match model's expected shape
            if len(x.shape) == 2:
                # If input is (batch_size, n_timesteps * n_features)
                n_samples = x.shape[0]
                x = x.reshape(n_samples, n_timesteps, n_features)
            elif len(x.shape) == 1:
                # If input is (n_timesteps * n_features,)
                x = x.reshape(1, n_timesteps, n_features)
            
            # Get model predictions
            predictions = self.regime_model.predict(x, verbose=0)
            return predictions[:, 0]  # Return first output for SHAP
            
        except Exception as e:
            logger.error(f"Error in regime prediction: {str(e)}")
            raise

    def _predict_trend(self, x: np.ndarray) -> np.ndarray:
        """Predict trend direction with proper shape handling"""
        try:
            # Get the number of features
            n_features = len(self.feature_columns)
            
            # Calculate the number of timesteps
            n_timesteps = 60  # Model expects 60 timesteps
            
            # Reshape input to match model's expected shape
            if len(x.shape) == 2:
                # If input is (batch_size, n_timesteps * n_features)
                n_samples = x.shape[0]
                x = x.reshape(n_samples, n_timesteps, n_features)
            elif len(x.shape) == 1:
                # If input is (n_timesteps * n_features,)
                x = x.reshape(1, n_timesteps, n_features)
            
            # Get model predictions
            predictions = self.transformer_model.predict(x, verbose=0)
            return predictions.flatten()  # Return flattened predictions
            
        except Exception as e:
            logger.error(f"Error in trend prediction: {str(e)}")
            raise

    def optimize_features(self, input_data: Union[str, np.ndarray], top_k: int = 15) -> np.ndarray:
        """Day trading focused feature optimization"""
        try:
            logger.info("Starting feature optimization for day trading")
            
            # Load data
            data = self._load_input_data(input_data)
            logger.info(f"Loaded data shape: {data.shape}")
            
            # Calculate total number of features across all timesteps
            n_samples, n_timesteps, n_features = data.shape
            total_features = n_timesteps * n_features
            required_evals = 2 * total_features + 1
            logger.info(f"Total features across timesteps: {total_features}")
            logger.info(f"Required evaluations: {required_evals}")
            
            # Calculate npermutations to ensure we have enough evaluations
            npermutations = int(np.ceil(required_evals / n_timesteps))
            logger.info(f"Number of timesteps: {n_timesteps}")
            logger.info(f"Using {npermutations} permutations per feature")
            logger.info(f"Total evaluations will be {npermutations * n_timesteps}")
            
            # Process data in batches to handle large datasets
            batch_size = 100  # Process 100 samples at a time
            all_regime_shap = []
            all_trend_shap = []
            
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch_data = data[i:batch_end]
                
                # Reshape batch data for SHAP computation
                batch_reshaped = batch_data.reshape(batch_end - i, n_timesteps * n_features)
                logger.info(f"Processing batch {i//batch_size + 1}, shape: {batch_reshaped.shape}")
                
                # Compute SHAP values for the batch
                regime_shap = self.regime_explainer.shap_values(batch_reshaped, npermutations=npermutations)
                trend_shap = self.trend_explainer.shap_values(batch_reshaped, npermutations=npermutations)
                
                # Handle list outputs from SHAP
                if isinstance(regime_shap, list):
                    regime_shap = regime_shap[0]
                if isinstance(trend_shap, list):
                    trend_shap = trend_shap[0]
                
                # Reshape SHAP values back to original dimensions
                regime_shap = regime_shap.reshape(batch_end - i, n_timesteps, n_features)
                trend_shap = trend_shap.reshape(batch_end - i, n_timesteps, n_features)
                
                all_regime_shap.append(regime_shap)
                all_trend_shap.append(trend_shap)
            
            # Combine all batches
            regime_shap = np.concatenate(all_regime_shap, axis=0)
            trend_shap = np.concatenate(all_trend_shap, axis=0)
            
            # Combine SHAP values with trading-specific weights
            importance = self._combine_shap_values(regime_shap, trend_shap)
            logger.info(f"Combined importance shape: {importance.shape}")
            
            # Apply trading-specific weights
            weights = self._trading_feature_weights()
            importance *= weights
            logger.info("Applied trading-specific weights")
            
            # Force include essential features
            essential_idx = [self.feature_columns.index(f) for f in self.essential_features]
            importance[essential_idx] += 1000
            logger.info("Added essential features to selection")
            
            # Select top features
            top_features = np.argsort(importance)[-top_k:]
            logger.info(f"Selected top {top_k} features")
            
            # Log selected features
            selected_features = [self.feature_columns[i] for i in top_features]
            logger.info(f"Selected features: {selected_features}")
            
            return top_features
            
        except Exception as e:
            logger.error(f"Feature optimization failed: {str(e)}")
            raise

    def _trading_feature_weights(self) -> np.ndarray:
        """Trading-specific feature weights"""
        weights = np.ones(len(self.feature_columns))
        
        # Higher weights for execution-critical features
        for i, feature in enumerate(self.feature_columns):
            if 'bid_ask_spread' in feature:
                weights[i] *= 2.0  # Critical for execution
            elif 'volume' in feature:
                weights[i] *= 1.5  # Important for liquidity
            elif 'vwap' in feature:
                weights[i] *= 1.3  # Price impact
            elif 'days_since_dividend' in feature or 'split_ratio' in feature:
                weights[i] *= 1.2  # Corporate actions
                
        logger.info(f"Applied trading weights: {dict(zip(self.feature_columns, weights))}")
        return weights

    def _combine_shap_values(self, regime_shap: np.ndarray, trend_shap: np.ndarray) -> np.ndarray:
        """Combine SHAP values from different models with trading-specific weights"""
        # Weight regime and trend predictions
        regime_weight = 0.4  # Less weight on regime in day trading
        trend_weight = 0.6   # More weight on trend
        
        # Ensure proper shape handling
        if isinstance(regime_shap, list):
            regime_shap = regime_shap[0]
        if isinstance(trend_shap, list):
            trend_shap = trend_shap[0]
        
        # Combine SHAP values
        combined = (
            regime_weight * np.abs(regime_shap).mean(axis=1).mean(axis=0) +
            trend_weight * np.abs(trend_shap).mean(axis=1).mean(axis=0)
        )
        
        logger.info(f"Combined SHAP values shape: {combined.shape}")
        return combined

    def _reshape_for_shap(self, data: np.ndarray) -> np.ndarray:
        """Reshape 3D data to 2D for SHAP computation with feature validation"""
        n_samples, n_timesteps, n_features = data.shape
        logger.info(f"Reshaping 3D data: samples={n_samples}, timesteps={n_timesteps}, features={n_features}")
        
        # Validate feature count before reshaping
        if n_features != len(self.feature_columns):
            raise ValueError(f"Input features ({n_features}) don't match expected features ({len(self.feature_columns)})")
        
        # Preserve feature order during reshape
        reshaped = data.reshape(n_samples * n_timesteps, n_features)
        logger.info(f"Reshaped data shape: {reshaped.shape}")
        return reshaped

    def _reshape_back_to_3d(self, data: np.ndarray, original_shape: tuple) -> np.ndarray:
        """Reshape 2D SHAP values back to 3D"""
        logger.info(f"Reshaping back to original shape: {original_shape}")
        reshaped = data.reshape(original_shape)
        logger.info(f"Reshaped back to 3D: {reshaped.shape}")
        return reshaped

    def _get_latest_regime_model(self):
        """Get the latest regime model path from S3"""
        s3 = boto3.client('s3')
        prefix = 'models/enhanced_v'
        logger.info(f"Searching for models in s3://{self.registry.bucket}/{prefix}")
        
        # List all objects with the prefix
        response = s3.list_objects_v2(
            Bucket=self.registry.bucket,
            Prefix=prefix
        )
        
        # Get all versioned models, excluding metadata files
        versions = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            if 'regime_model.h5' in key and not key.endswith('.metadata'):
                versions.append(key)
                logger.debug(f"Found model: {key}")
        
        if not versions:
            logger.error(f"No models found in s3://{self.registry.bucket}/{prefix}")
            raise ValueError("No versioned models found in S3")
            
        # Get the latest version (sort by timestamp in filename)
        latest_version = sorted(versions)[-1]
        logger.info(f"Using latest model: {latest_version}")
            
        return f"s3://{self.registry.bucket}/{latest_version}"

    def _load_model_from_s3(self, s3_path: str) -> tf.keras.Model:
        """Load model from S3 into memory"""
        s3 = boto3.client('s3')
        
        try:
            # If path doesn't start with s3://, treat as local path
            if not s3_path.startswith('s3://'):
                logger.info(f"Loading model from local path: {s3_path}")
                return tf.keras.models.load_model(s3_path)
            
            # Parse S3 path
            bucket = s3_path.split('/')[2]
            key = '/'.join(s3_path.split('/')[3:])
            
            logger.info(f"Downloading model from s3://{bucket}/{key}")
            
            # Download model to memory
            response = s3.get_object(Bucket=bucket, Key=key)
            model_data = BytesIO(response['Body'].read())
            
            # Get the raw bytes
            raw_data = model_data.getvalue()
            logger.info(f"Downloaded model size: {len(raw_data)} bytes")
            
            if len(raw_data) == 0:
                raise ValueError("Downloaded model is empty")
            
            # Create a temporary file with proper HDF5 extension
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
                # Write the raw bytes directly
                temp_file.write(raw_data)
                temp_file.flush()  # Ensure all data is written
                temp_path = temp_file.name
                logger.info(f"Created temporary file at {temp_path}")
            
            try:
                # Verify file exists and has content
                if not os.path.exists(temp_path):
                    raise FileNotFoundError(f"Temporary file not created at {temp_path}")
                
                file_size = os.path.getsize(temp_path)
                logger.info(f"Temporary file size: {file_size} bytes")
                
                if file_size == 0:
                    raise ValueError("Temporary file is empty")
                
                # Try to read the first few bytes to verify it's a valid HDF5 file
                with open(temp_path, 'rb') as f:
                    header = f.read(8)
                    logger.info(f"File header (hex): {header.hex()}")
                
                # Load model from temporary file
                logger.info("Loading model from temporary file...")
                model = tf.keras.models.load_model(temp_path)
                logger.info("Model loaded successfully")
                return model
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                # Log the first few bytes of the file to help diagnose the issue
                try:
                    with open(temp_path, 'rb') as f:
                        header = f.read(32)
                        logger.error(f"File header (hex): {header.hex()}")
                except Exception as read_error:
                    logger.error(f"Could not read file header: {str(read_error)}")
                raise
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                    logger.info("Temporary file cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up temporary file: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _load_transformer_model(self) -> tf.keras.Model:
        """Load transformer model from S3"""
        try:
            s3 = boto3.client('s3')
            prefix = 'models/enhanced_v'
            
            # List all objects with the prefix
            response = s3.list_objects_v2(
                Bucket=self.registry.bucket,
                Prefix=prefix
            )
            
            # Get all versioned models, excluding metadata files
            versions = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                if 'transformer_trend.h5' in key and not key.endswith('.metadata'):
                    versions.append(key)
                    logger.debug(f"Found transformer model: {key}")
            
            if not versions:
                raise ValueError("No transformer models found in S3")
                
            # Get the latest version
            latest_version = sorted(versions)[-1]
            logger.info(f"Using latest transformer model: {latest_version}")
            
            # Load model from S3
            s3_path = f"s3://{self.registry.bucket}/{latest_version}"
            return self._load_model_from_s3(s3_path)
            
        except Exception as e:
            logger.error(f"Error loading transformer model: {str(e)}")
            raise

    def _validate_background_samples(self, samples: np.ndarray) -> bool:
        """Validate background samples for trading suitability"""
        try:
            # Check for minimum spread
            spread_idx = self.feature_columns.index('bid_ask_spread')
            min_spread = np.min(samples[:, :, spread_idx])
            if min_spread < self.min_liquidity_threshold:
                logger.warning(f"Background samples contain low spreads: {min_spread} (threshold: {self.min_liquidity_threshold})")
                logger.warning(f"Spread statistics - min: {min_spread}, max: {np.max(samples[:, :, spread_idx])}, mean: {np.mean(samples[:, :, spread_idx])}")
                return False
            
            # Check for volume - using mean instead of min to be more lenient
            volume_idx = self.feature_columns.index('volume')
            mean_volume = np.mean(samples[:, :, volume_idx])
            if mean_volume < 100:  # Lowered threshold and using mean
                logger.warning(f"Background samples have low average volume: {mean_volume} (threshold: 100)")
                logger.warning(f"Volume statistics - min: {np.min(samples[:, :, volume_idx])}, max: {np.max(samples[:, :, volume_idx])}, mean: {mean_volume}")
                return False
            
            # Check for price data validity
            price_idx = self.feature_columns.index('close')
            invalid_prices = np.sum(samples[:, :, price_idx] <= 0)
            if invalid_prices > 0:
                logger.warning(f"Background samples contain {invalid_prices} invalid prices (zero or negative)")
                logger.warning(f"Price statistics - min: {np.min(samples[:, :, price_idx])}, max: {np.max(samples[:, :, price_idx])}, mean: {np.mean(samples[:, :, price_idx])}")
                return False
            
            # Check for data completeness
            nan_count = np.sum(np.isnan(samples))
            if nan_count > 0:
                logger.warning(f"Background samples contain {nan_count} NaN values")
                # Log which features have NaN values
                for i, feature in enumerate(self.feature_columns):
                    nan_feature_count = np.sum(np.isnan(samples[:, :, i]))
                    if nan_feature_count > 0:
                        logger.warning(f"Feature '{feature}' has {nan_feature_count} NaN values")
                return False
            
            logger.info(f"Background samples validated successfully with shape {samples.shape}")
            logger.info(f"Sample statistics:")
            logger.info(f"- Spread: min={np.min(samples[:, :, spread_idx]):.6f}, max={np.max(samples[:, :, spread_idx]):.6f}, mean={np.mean(samples[:, :, spread_idx]):.6f}")
            logger.info(f"- Volume: min={np.min(samples[:, :, volume_idx]):.0f}, max={np.max(samples[:, :, volume_idx]):.0f}, mean={mean_volume:.0f}")
            logger.info(f"- Price: min={np.min(samples[:, :, price_idx]):.2f}, max={np.max(samples[:, :, price_idx]):.2f}, mean={np.mean(samples[:, :, price_idx]):.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating background samples: {str(e)}")
            return False

    def _prepare_background(self, data: Optional[pd.DataFrame], n_samples: int) -> np.ndarray:
        """Prepare background data with trading-specific sampling"""
        try:
            if data is None:
                logger.info(f"Loading background data for {self.ticker}")
                return self._load_production_background(n_samples)
            
            logger.info(f"Preparing background data with {n_samples} samples")
            # Use the existing data loader instance
            sequences = self.data_loader.create_sequences(data)
            logger.info(f"Created sequences shape: {sequences.shape}")
            
            # Sample background data with trading-specific considerations
            if len(sequences) > n_samples:
                # Ensure we have enough samples from different market conditions
                indices = np.random.choice(len(sequences), n_samples, replace=False)
                background = sequences[indices]
                
                # Validate sample quality
                if not self._validate_background_samples(background):
                    logger.warning(f"Background samples validation failed for {self.ticker}")
                    # Instead of loading production data, try to filter the current data
                    logger.info("Attempting to filter current data for better quality samples")
                    df = data.copy()
                    df = self._filter_trading_data(df)
                    sequences = self.data_loader.create_sequences(df)
                    
                    if len(sequences) > n_samples:
                        indices = np.random.choice(len(sequences), n_samples, replace=False)
                        background = sequences[indices]
                    else:
                        background = sequences
                    
                    # Final validation check
                    if not self._validate_background_samples(background):
                        logger.error("Failed to generate valid background samples after filtering")
                        raise ValueError("Could not generate valid background samples")
            else:
                background = sequences
                
            logger.info(f"Final background shape: {background.shape}")
            return background
            
        except Exception as e:
            logger.error(f"Error preparing background data: {str(e)}")
            raise

    def _load_production_background(self, n_samples: int) -> np.ndarray:
        """Load real market data from S3 with trading-specific filtering"""
        try:
            logger.info(f"Loading production background data for {self.ticker} with {n_samples} samples")
            
            # Use the existing data loader instance
            df = self.data_loader.load_ticker_data(self.ticker)
            if df is None or df.empty:
                raise ValueError(f"No data available for {self.ticker}")
            
            # Filter for trading hours and sufficient liquidity
            df = self._filter_trading_data(df)
            
            # Create sequences using the existing data loader
            sequences = self.data_loader.create_sequences(df)
            logger.info(f"Created sequences shape: {sequences.shape}")
            
            # Sample recent data for better market representation
            n_samples = min(n_samples, len(sequences))
            background = sequences[-n_samples:]
            
            # Validate the production background
            if not self._validate_background_samples(background):
                raise ValueError("Production background samples failed validation")
            
            logger.info(f"Final background shape: {background.shape}")
            return background
            
        except Exception as e:
            logger.error(f"Error loading production background: {str(e)}")
            raise

    def _filter_trading_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data for trading suitability"""
        try:
            # Filter for minimum spread
            df = df[df['bid_ask_spread'] >= self.min_liquidity_threshold]
            
            # Filter for minimum volume
            df = df[df['volume'] >= 1000]
            
            # Filter for reasonable price impact
            df = df[df['vwap'] > 0]
            
            # Remove outliers
            for col in ['bid_ask_spread', 'volume']:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                df = df[df[col] <= q3 + 1.5 * iqr]
                df = df[df[col] >= q1 - 1.5 * iqr]
            
            logger.info(f"Filtered data shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error filtering trading data: {str(e)}")
            return df

    def _load_input_data(self, input_data: Union[str, np.ndarray]) -> np.ndarray:
        """Load and validate input data"""
        try:
            if isinstance(input_data, str):
                logger.info(f"Loading input data from {input_data}")
                if input_data.endswith('.npz'):
                    data = np.load(input_data)
                    if 'X' in data:
                        data = data['X']
                        logger.info(f"Loaded data shape: {data.shape}")
                        logger.info(f"Loaded data features: {data.shape[-1]}")
                    else:
                        raise ValueError(f"No 'X' array found in {input_data}")
                else:
                    data = joblib.load(input_data)
                    logger.info(f"Loaded data shape: {data.shape}")
                    logger.info(f"Loaded data features: {data.shape[-1]}")
            else:
                data = input_data
                logger.info(f"Using pre-loaded data shape: {data.shape}")
                logger.info(f"Using pre-loaded data features: {data.shape[-1]}")
            
            # Validate data
            if len(data.shape) != 3:
                raise ValueError(f"Input data must be 3D, got {len(data.shape)}D")
            
            if data.shape[-1] != len(self.feature_columns):
                raise ValueError(f"Input features ({data.shape[-1]}) don't match expected features ({len(self.feature_columns)})")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading input data: {str(e)}")
            raise