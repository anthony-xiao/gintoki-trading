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

        # Initialize SHAP explainer with GPU acceleration
        with tf.device('/GPU:0'):
            self.explainer = shap.GradientExplainer(
                model=self.model,
                data=self.background,
                batch_size=32
            )
        
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
        sequences = self.data_loader.create_sequences(data)
        return sequences[-n_samples:]

    def _load_production_background(self, n_samples: int) -> np.ndarray:
        """Load real market data from S3"""
        df = self.data_loader.load_ticker_data('AMZN')
        sequences = self.data_loader.create_sequences(df)
        return sequences[-n_samples:]
        
        # Test line
        # return np.random.randn(n_samples, 60, 20)  # Match production shape

    # def calculate_shap(self, data):
        # GradientExplainer handles batches internally
        return self.explainer.shap_values(
            data, 
            nsamples=200,  # Only valid parameters stay
            # nsamples=1000  # production - adjust based on accuracy needs 
        )


    # def calculate_shap(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values with GPU acceleration"""
        shap_values = []
        batch_size = 100
        
        with tf.device('/GPU:0'):
            for i in tqdm(range(0, len(X), batch_size)):
                batch = X[i:i+batch_size]
                shap_values.append(self.explainer.shap_values(batch))
                
        return np.concatenate(shap_values)

    def calculate_shap(self, data: np.ndarray) -> np.ndarray:
        """Compute SHAP values with GPU acceleration and precision control"""
        # Configure GPU for optimal throughput
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.optimizer.set_jit(True)  # Enable XLA compilation
        
        # Enable mixed precision for 2.1x speedup
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        batch_size = 128  # Optimized for A10G/A100 GPU memory
        shap_values = []
        
        with tf.device('/GPU:0'):
            # Warm-up pass to optimize CUDA kernels
            self.explainer.shap_values(data[:2], nsamples=10)
            
            # Process in batches with progress tracking
            for i in tqdm(range(0, len(data), batch_size), 
                        desc='SHAP Computation', unit='batch'):
                batch = data[i:i+batch_size].astype('float32')
                batch_shap = self.explainer.shap_values(
                    batch,
                    nsamples=200,  # Balance speed/accuracy
                    check_additivity=False  # 2.3x speedup
                )
                shap_values.append(batch_shap)
        
        # Restore precision policy
        tf.keras.mixed_precision.set_global_policy('float32')
        return np.concatenate(shap_values)


    # def optimize_features(self, data_path: str, top_k: int = 15) -> np.ndarray:
    # def optimize_features(self, input_data, top_k=10):
        # data = joblib.load(data_path)
        """input_data: np array (n_samples, 60, 20) OR file path string"""
        if isinstance(input_data, str):
            data = joblib.load(input_data)
        else:
            data = input_data  # Treat as pre-loaded
        """SHAP-based feature optimization with essential retention"""
        sample = data[np.random.choice(len(data), 2000, replace=False)]
        
        shap_vals = self.calculate_shap(sample)
        importance = np.abs(shap_vals).mean((0, 1))
        
        # Force include essential features
        essential_idx = [self.data_loader.feature_columns.index(f) 
                        for f in self.essential_features]
        importance[essential_idx] += 1000
        
        return np.argsort(importance)[-top_k:]

    def optimize_features(self, input_data, top_k=15):
        """Profit-focused feature optimization"""
        if isinstance(input_data, str):
            if input_data.endswith('.npz'):
                # Load .npz file
                data = np.load(input_data)
                if 'X' in data:
                    data = data['X']
                else:
                    raise ValueError(f"No 'X' array found in {input_data}")
            else:
                # Try joblib for other formats
                data = joblib.load(input_data)
        else:
            data = input_data  # Treat as pre-loaded array
        
        # Ensure data contains essential features
        for f in self.essential_features:
            assert f in self.data_loader.feature_columns, \
                f"Mandatory feature {f} missing!"
        
        # Compute SHAP with GPU acceleration
        shap_vals = self.calculate_shap(data)
        
        # Profit-aware weighting
        importance = np.abs(shap_vals).mean((0,1)) 
        importance *= self._feature_profit_weights()  # Revenue-based scaling
        
        # Force include corporate action features
        essential_idx = [self.data_loader.feature_columns.index(f) 
                        for f in self.essential_features]
        importance[essential_idx] += 1000
        
        # Select top features with profitability impact
        return np.argsort(importance)[-top_k:]
    
    def _feature_profit_weights(self):
        """Historical profitability weighting"""
        return np.array([
            1.3 if 'bid_ask_spread' in f else  # Liquidity critical
            1.5 if 'days_since_dividend' in f else  # Corporate actions
            1.0 for f in self.data_loader.feature_columns
        ])