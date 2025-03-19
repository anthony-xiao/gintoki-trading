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
from sklearn.model_selection import train_test_split


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
        
        # Create a wrapper function for the model that handles 3D input
        def model_predict(x):
            # Reshape input if needed (for background samples)
            if len(x.shape) == 2:
                x = x.reshape(-1, self.model.input_shape[1], self.model.input_shape[2])
            return self.model.predict(x, verbose=0)
        
        # Use provided background data or load it
        if background_data is not None:
            self.background = self._prepare_background(background_data, background_samples)
        else:
            self.background = self._load_production_background(background_samples)
        
        # Initialize SHAP explainer with KernelExplainer
        self.explainer = shap.KernelExplainer(
            model_predict,
            self.background
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
        """Prepare background data from provided DataFrame with efficient sampling"""
        sequences = self.data_loader.create_sequences(data)
        
        # Use stratified sampling if possible
        if 'regime' in data.columns:
            labels = data['regime'].values[self.model.input_shape[1]:]
            _, _, sequences, _ = train_test_split(
                sequences, labels,
                test_size=1 - (n_samples/len(sequences)),
                stratify=labels,
                random_state=42
            )
        else:
            # Random sampling if no labels available
            indices = np.random.choice(len(sequences), size=min(n_samples, len(sequences)), replace=False)
            sequences = sequences[indices]
        
        # Reshape background data to 2D for SHAP
        n_samples, seq_len, n_features = sequences.shape
        return sequences.reshape(n_samples, seq_len * n_features)

    def _load_production_background(self, n_samples: int) -> np.ndarray:
        """Load real market data from S3 with efficient sampling"""
        df = self.data_loader.load_ticker_data('AMZN')
        return self._prepare_background(df, n_samples)

    def calculate_shap(self, data: np.ndarray) -> np.ndarray:
        """Compute SHAP values with efficient batching"""
        batch_size = 128  # Optimized for memory
        shap_values = []
        
        # Get original shape
        n_samples, seq_len, n_features = data.shape
        
        # Reshape data to 2D for SHAP
        data_2d = data.reshape(n_samples, seq_len * n_features)
        
        # Process in batches with progress tracking
        for i in tqdm(range(0, len(data_2d), batch_size), 
                    desc='SHAP Computation', unit='batch'):
            batch = data_2d[i:i+batch_size]
            batch_shap = self.explainer.shap_values(
                batch,
                nsamples=100,  # Balance between speed and accuracy
                silent=True
            )
            shap_values.append(batch_shap)
        
        # Combine all SHAP values and reshape back to 3D
        shap_values = np.concatenate(shap_values)
        return shap_values.reshape(n_samples, seq_len, n_features)

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
        
        # Compute SHAP with efficient batching
        shap_vals = self.calculate_shap(data)
        
        # Average SHAP values across time dimension
        importance = np.abs(shap_vals).mean(axis=1).mean(axis=0)  # Average across samples and time
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