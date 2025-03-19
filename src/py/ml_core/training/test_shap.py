import os
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.py.ml_core.shap_optimizer import EnhancedSHAPOptimizer
from src.py.ml_core.data_loader import EnhancedDataLoader
import gc
import boto3
from io import BytesIO
from botocore.config import Config
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_s3_key(key: str, bucket: str, s3_client: boto3.client) -> pd.DataFrame:
    """Process a single S3 key in parallel"""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        bio = BytesIO(response['Body'].read())
        df = pd.read_parquet(bio)
        
        # Downsample large files
        if len(df) > 1_000_000:
            df = df.sample(1_000_000)
            
        return df
    except Exception as e:
        logger.error(f"Error processing {key}: {str(e)}")
        return pd.DataFrame()

def load_ticker_data(ticker: str, data_loader: EnhancedDataLoader) -> np.ndarray:
    """Load and process data for a single ticker with parallel S3 loading"""
    try:
        # Initialize S3 client with optimized config
        s3 = boto3.client('s3',
                        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-west-2'),
                        config=Config(signature_version='s3v4', max_pool_connections=50))
        
        # Get all Parquet keys for the ticker
        prefix = f'historical/{ticker}/aggregates/minute/'
        keys = []
        paginator = s3.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=data_loader.bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('/') or not key.lower().endswith('.parquet'):
                    continue
                keys.append(key)
        
        logger.info(f"Found {len(keys)} files to process for {ticker}")
        
        # Process files in parallel
        dfs = []
        with ThreadPoolExecutor(max_workers=30) as executor:
            future_to_key = {
                executor.submit(process_s3_key, key, data_loader.bucket, s3): key 
                for key in keys
            }
            
            for future in tqdm(future_to_key, desc=f"Loading {ticker} data"):
                key = future_to_key[future]
                try:
                    df = future.result()
                    if not df.empty:
                        dfs.append(df)
                except Exception as e:
                    logger.error(f"Failed to process {key}: {str(e)}")
        
        if not dfs:
            logger.warning(f"No data loaded for {ticker}")
            return None
            
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=False).sort_index()
        logger.info(f"Combined data shape for {ticker}: {combined_df.shape}")
        
        # Create sequences
        sequences = data_loader.create_sequences(combined_df)
        logger.info(f"Created sequences for {ticker} with shape {sequences.shape}")
        
        # Clear memory
        del dfs
        del combined_df
        gc.collect()
        
        return sequences
        
    except Exception as e:
        logger.error(f"Error loading {ticker}: {str(e)}")
        return None

def test_shap_optimization():
    try:
        logger.info("🚀 Starting SHAP optimization test")
        
        # Initialize components
        data_loader = EnhancedDataLoader()
        optimizer = EnhancedSHAPOptimizer(background_samples=100)
        
        # Define test tickers
        test_tickers = ['SMCI']
        
        # Load data in parallel
        logger.info("📦 Loading sample data in parallel...")
        all_sequences = []
        
        with ThreadPoolExecutor(max_workers=30) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(load_ticker_data, ticker, data_loader): ticker 
                for ticker in test_tickers
            }
            
            # Process completed tasks with progress bar
            for future in tqdm(future_to_ticker, desc="Loading tickers"):
                ticker = future_to_ticker[future]
                try:
                    sequences = future.result()
                    if sequences is not None:
                        all_sequences.append(sequences)
                        logger.info(f"✅ Successfully loaded {ticker}")
                except Exception as e:
                    logger.error(f"❌ Failed to load {ticker}: {str(e)}")
        
        if not all_sequences:
            raise ValueError("No data loaded for any tickers")
        
        # Combine all sequences
        combined_sequences = np.concatenate(all_sequences)
        logger.info(f"📊 Combined sequences shape: {combined_sequences.shape}")
        
        # Clear memory
        del all_sequences
        gc.collect()
        
        # Save sequences for testing
        np.savez('test_data.npz', X=combined_sequences)
        logger.info("💾 Saved test data to test_data.npz")
        
        # Run SHAP optimization
        logger.info("🎯 Running SHAP optimization...")
        top_features = optimizer.optimize_features('test_data.npz', top_k=15)
        
        # Print results
        feature_names = data_loader.feature_columns
        logger.info("\n📊 Top 15 Features:")
        for idx in reversed(top_features):
            logger.info(f"{feature_names[idx]}")
            
        logger.info("✅ SHAP optimization test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    test_shap_optimization() 