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

logger = logging.getLogger(__name__)

class EnhancedSHAPOptimizer:
    def __init__(self, model_path=None, background_samples=1000, background_data=None):
        """Initialize SHAP optimizer with latest S3 model"""
        self.registry = EnhancedModelRegistry()
        self.data_loader = EnhancedDataLoader()
        
        # Log feature columns
        self.feature_columns = self.data_loader.feature_columns
        logger.info(f"Available feature columns: {self.feature_columns}")
        logger.info(f"Number of features: {len(self.feature_columns)}")
        
        # Get latest regime model from S3
        if model_path is None:
            model_path = self._get_latest_regime_model()
        
        # Download and load model from S3
        self.model = self._load_model_from_s3(model_path)
        self.input_name = self.model.layers[0].name
        
        # Use provided background data or load it
        if background_data is not None:
            self.background = self._prepare_background(background_data, background_samples)
        else:
            self.background = self._load_production_background(background_samples)

        logger.info(f"Background data shape: {self.background.shape}")
        logger.info(f"Background data features: {self.background.shape[-1]}")
        
        # Validate background data features
        if self.background.shape[-1] != len(self.feature_columns):
            raise ValueError(f"Background data features ({self.background.shape[-1]}) don't match feature columns ({len(self.feature_columns)})")
        
        # Prepare background data for SHAP
        background_2d = self.background.reshape(-1, len(self.feature_columns))
        logger.info(f"Reshaped background data shape: {background_2d.shape}")
        
        # Initialize SHAP explainer with KernelExplainer
        logger.info("Initializing KernelExplainer...")
        self.explainer = shap.KernelExplainer(
            model=lambda x: self._predict_3d(x)[:, 0],  # Only use first output for SHAP
            data=background_2d,
            link="identity"  # Use identity link for better numerical stability
        )
        logger.info("KernelExplainer initialized successfully")
        
        self.essential_features = ['days_since_dividend', 'split_ratio', 'bid_ask_spread']
        logger.info(f"Essential features: {self.essential_features}")

    def _predict_3d(self, x):
        """Helper function to handle 3D predictions"""
        # If input is 2D, reshape to 3D
        if len(x.shape) == 2:
            n_samples = x.shape[0]
            n_features = x.shape[1]
            n_timesteps = self.background.shape[1]  # Get timesteps from background
            x = x.reshape(n_samples // n_timesteps, n_timesteps, n_features)
            logger.debug(f"Reshaped 2D input to 3D: {x.shape}")
        
        # Validate feature count
        if x.shape[-1] != len(self.feature_columns):
            raise ValueError(f"Input features ({x.shape[-1]}) don't match expected features ({len(self.feature_columns)})")
        
        # Ensure input matches model's expected shape
        if x.shape[1:] != self.background.shape[1:]:
            raise ValueError(f"Input shape {x.shape} doesn't match expected shape {self.background.shape}")
            
        return self.model.predict(x, verbose=0)

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

    def _load_model_from_s3(self, s3_path):
        """Load model from S3 into memory"""
        s3 = boto3.client('s3')
        
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

    def _prepare_background(self, data: pd.DataFrame, n_samples: int) -> np.ndarray:
        """Prepare background data from provided DataFrame"""
        logger.info(f"Preparing background data with {n_samples} samples")
        sequences = self.data_loader.create_sequences(data)
        logger.info(f"Created sequences shape: {sequences.shape}")
        
        # Sample background data
        if len(sequences) > n_samples:
            # Randomly sample n_samples sequences
            indices = np.random.choice(len(sequences), n_samples, replace=False)
            background = sequences[indices]
        else:
            background = sequences
            
        logger.info(f"Final background shape: {background.shape}")
        return background

    def _load_production_background(self, n_samples: int) -> np.ndarray:
        """Load real market data from S3"""
        logger.info(f"Loading production background data with {n_samples} samples")
        # The ticker should be passed in from train.py's args.tickers
        # But it seems the ticker attribute is not being set properly
        if not hasattr(self, 'ticker') or self.ticker is None:
            logger.warning("No ticker specified, defaulting to SMCI")
            self.ticker = 'SMCI'
        df = self.data_loader.load_ticker_data(self.ticker)
        sequences = self.data_loader.create_sequences(df)
        logger.info(f"Created sequences shape: {sequences.shape}")
        
        # Keep 3D structure
        n_samples = min(n_samples, len(sequences))
        background = sequences[-n_samples:]
        logger.info(f"Final background shape: {background.shape}")
        return background

    def calculate_shap(self, data: np.ndarray) -> np.ndarray:
        """Compute SHAP values with dimension validation"""
        logger.info(f"Starting SHAP calculation for data shape: {data.shape}")
        
        # Validate input dimensions
        if len(data.shape) != 3:
            raise ValueError(f"Input data must be 3D, got {len(data.shape)}D")
        
        # Process in smaller batches with progress tracking
        batch_size = 16
        shap_values = []
        
        try:
            for i in tqdm(range(0, len(data), batch_size), 
                        desc='SHAP Computation', unit='batch'):
                batch = data[i:i+batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}, shape: {batch.shape}")
                
                # Reshape batch for SHAP computation
                batch_2d = batch.reshape(-1, len(self.feature_columns))
                logger.debug(f"Reshaped batch shape: {batch_2d.shape}")
                
                # Compute SHAP values
                batch_shap = self.explainer.shap_values(
                    batch_2d,
                    nsamples=50,
                    silent=True
                )
                
                # Handle multi-output format
                if isinstance(batch_shap, list):
                    batch_shap = batch_shap[0]  # Take first output
                
                # Reshape back to original batch shape
                batch_shap_3d = batch_shap.reshape(batch.shape)
                logger.debug(f"Reshaped SHAP values shape: {batch_shap_3d.shape}")
                shap_values.append(batch_shap_3d)
            
            # Combine all batches
            final_shap = np.concatenate(shap_values, axis=0)
            logger.info(f"Final SHAP values shape: {final_shap.shape}")
            return final_shap
        
        except Exception as e:
            logger.error(f"SHAP calculation failed: {str(e)}")
            raise

    def optimize_features(self, input_data, top_k=15):
        """Profit-focused feature optimization with enhanced logging"""
        logger.info("Starting feature optimization")
        
        try:
            # Load and validate input data
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
            
            # Validate essential features
            for f in self.essential_features:
                if f not in self.data_loader.feature_columns:
                    raise ValueError(f"Mandatory feature {f} missing!")
            
            # Compute SHAP values
            logger.info("Computing SHAP values...")
            shap_vals = self.calculate_shap(data)
            logger.info(f"SHAP values computed, shape: {shap_vals.shape}")
            
            # Calculate feature importance (average across time steps)
            importance = np.abs(shap_vals).mean(axis=1).mean(axis=0)
            logger.info(f"Computed feature importance shape: {importance.shape}")
            logger.info(f"Feature importance values: {importance}")
            
            # Apply profit weights
            weights = self._feature_profit_weights()
            importance *= weights
            logger.info("Applied profit weights to importance scores")
            logger.info(f"Weighted importance values: {importance}")
            
            # Force include essential features
            essential_idx = [self.data_loader.feature_columns.index(f) 
                           for f in self.essential_features]
            importance[essential_idx] += 1000
            logger.info("Added essential features to selection")
            
            # Select top features
            top_features = np.argsort(importance)[-top_k:]
            logger.info(f"Selected top {top_k} features")
            
            return top_features
            
        except Exception as e:
            logger.error(f"Error in feature optimization: {str(e)}")
            raise
    
    def _feature_profit_weights(self):
        """Historical profitability weighting"""
        return np.array([
            1.3 if 'bid_ask_spread' in f else  # Liquidity critical
            1.5 if 'days_since_dividend' in f else  # Corporate actions
            1.0 for f in self.data_loader.feature_columns
        ])