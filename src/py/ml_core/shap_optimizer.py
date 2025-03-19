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
        
        # Reshape background data for SHAP
        self.background_2d = self._reshape_for_shap(self.background)
        logger.info(f"Reshaped background data shape: {self.background_2d.shape}")
        
        # Initialize SHAP explainer with KernelExplainer
        logger.info("Initializing KernelExplainer...")
        self.explainer = shap.KernelExplainer(
            model=lambda x: self.model.predict(x, verbose=0),
            data=self.background_2d
        )
        logger.info("KernelExplainer initialized successfully")
        
        self.essential_features = ['days_since_dividend', 'split_ratio', 'bid_ask_spread']

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
        
        # Keep 3D structure for KernelExplainer
        n_samples = min(n_samples, len(sequences))
        background = sequences[-n_samples:]
        logger.info(f"Final background shape: {background.shape}")
        return background

    def _load_production_background(self, n_samples: int) -> np.ndarray:
        """Load real market data from S3"""
        logger.info(f"Loading production background data with {n_samples} samples")
        df = self.data_loader.load_ticker_data('AMZN')
        sequences = self.data_loader.create_sequences(df)
        logger.info(f"Created sequences shape: {sequences.shape}")
        
        # Keep 3D structure for KernelExplainer
        n_samples = min(n_samples, len(sequences))
        background = sequences[-n_samples:]
        logger.info(f"Final background shape: {background.shape}")
        return background

    def _reshape_for_shap(self, data: np.ndarray) -> np.ndarray:
        """Reshape 3D data (samples, time_steps, features) to 2D for SHAP"""
        n_samples, n_timesteps, n_features = data.shape
        # Reshape to (samples * timesteps, features)
        return data.reshape(-1, n_features)

    def _reshape_back_to_3d(self, data: np.ndarray, original_shape: tuple) -> np.ndarray:
        """Reshape 2D SHAP values back to 3D"""
        return data.reshape(original_shape)

    def calculate_shap(self, data: np.ndarray) -> np.ndarray:
        """Compute SHAP values with detailed logging and error handling"""
        logger.info(f"Starting SHAP calculation for data shape: {data.shape}")
        
        # Store original shape for later reshaping
        original_shape = data.shape
        
        # Reshape data for SHAP
        data_2d = self._reshape_for_shap(data)
        logger.info(f"Reshaped data for SHAP calculation: {data_2d.shape}")
        
        # Verify feature count matches background data
        if data_2d.shape[1] != self.background_2d.shape[1]:
            raise ValueError(f"Feature count mismatch: input data has {data_2d.shape[1]} features, "
                           f"background data has {self.background_2d.shape[1]} features")
        
        # Process in batches with progress tracking
        batch_size = 32  # Smaller batch size for KernelExplainer
        shap_values = []
        
        try:
            for i in tqdm(range(0, len(data_2d), batch_size), 
                        desc='SHAP Computation', unit='batch'):
                batch = data_2d[i:i+batch_size].astype('float32')
                logger.debug(f"Processing batch {i//batch_size + 1}, shape: {batch.shape}")
                
                try:
                    batch_shap = self.explainer.shap_values(
                        batch,
                        nsamples=100,  # Number of samples for background distribution
                        silent=True  # Suppress progress bars
                    )
                    
                    # KernelExplainer returns a list of arrays for each output
                    if isinstance(batch_shap, list):
                        batch_shap = batch_shap[0]  # Take first output
                    
                    shap_values.append(batch_shap)
                    logger.debug(f"Successfully computed SHAP for batch {i//batch_size + 1}")
                    
                except Exception as e:
                    logger.error(f"Error computing SHAP for batch {i//batch_size + 1}: {str(e)}")
                    raise
            
            # Combine all SHAP values
            final_shap_2d = np.concatenate(shap_values)
            logger.info(f"Successfully computed SHAP values, shape: {final_shap_2d.shape}")
            
            # Reshape back to original 3D structure
            final_shap_3d = self._reshape_back_to_3d(final_shap_2d, original_shape)
            logger.info(f"Reshaped SHAP values back to original shape: {final_shap_3d.shape}")
            
            return final_shap_3d
            
        except Exception as e:
            logger.error(f"Critical error in SHAP calculation: {str(e)}")
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
                    else:
                        raise ValueError(f"No 'X' array found in {input_data}")
                else:
                    data = joblib.load(input_data)
                    logger.info(f"Loaded data shape: {data.shape}")
            else:
                data = input_data
                logger.info(f"Using pre-loaded data shape: {data.shape}")
            
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
            
            # Apply profit weights
            weights = self._feature_profit_weights()
            importance *= weights
            logger.info("Applied profit weights to importance scores")
            
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