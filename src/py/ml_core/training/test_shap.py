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
        
        logger.debug(f"Loaded data from {key} with shape: {df.shape}")
        logger.debug(f"Columns in loaded data: {df.columns.tolist()}")
        
        # Downsample large files
        if len(df) > 1_000_000:
            df = df.sample(1_000_000)
            logger.debug(f"Downsampled to shape: {df.shape}")
            
        return df
    except Exception as e:
        logger.error(f"Error processing {key}: {str(e)}")
        return pd.DataFrame()

def process_quotes(ticker: str, bucket: str, s3_client: boto3.client) -> pd.DataFrame:
    """Process quote data in parallel"""
    try:
        prefix = f'historical/{ticker}/quotes/'
        keys = []
        paginator = s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('/') or not key.lower().endswith('.parquet'):
                    continue
                keys.append(key)
        
        if not keys:
            logger.warning(f"No quote data found for {ticker}")
            return pd.DataFrame()
            
        logger.info(f"Found {len(keys)} quote files for {ticker}")
            
        # Process quote files in parallel
        dfs = []
        with ThreadPoolExecutor(max_workers=30) as executor:
            future_to_key = {
                executor.submit(process_s3_key, key, bucket, s3_client): key 
                for key in keys
            }
            
            for future in tqdm(future_to_key, desc=f"Loading {ticker} quotes"):
                key = future_to_key[future]
                try:
                    df = future.result()
                    if not df.empty:
                        dfs.append(df)
                        logger.debug(f"Successfully processed quote file: {key}")
                except Exception as e:
                    logger.error(f"Failed to process quote {key}: {str(e)}")
        
        if not dfs:
            return pd.DataFrame()
            
        # Combine and process quotes
        quotes_df = pd.concat(dfs, ignore_index=False).sort_index()
        logger.info(f"Combined quotes shape: {quotes_df.shape}")
        logger.info(f"Quote columns: {quotes_df.columns.tolist()}")
        
        # Resample to 1-minute bars
        quotes_df = quotes_df.resample('1min').agg({
            'bid_price': 'mean',
            'ask_price': 'mean',
            'bid_size': 'sum',
            'ask_size': 'sum'
        })
        logger.info(f"Resampled quotes shape: {quotes_df.shape}")
        
        # Calculate spread features
        quotes_df['bid_ask_spread'] = quotes_df['ask_price'] - quotes_df['bid_price']
        quotes_df['mid_price'] = (quotes_df['ask_price'] + quotes_df['bid_price']) / 2
        logger.info(f"Final quotes shape with spread features: {quotes_df.shape}")
        logger.info(f"Final quote columns: {quotes_df.columns.tolist()}")
        
        return quotes_df
        
    except Exception as e:
        logger.error(f"Error processing quotes for {ticker}: {str(e)}")
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
                        logger.debug(f"Successfully processed file: {key}")
                except Exception as e:
                    logger.error(f"Failed to process {key}: {str(e)}")
        
        if not dfs:
            logger.warning(f"No data loaded for {ticker}")
            return None
            
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=False).sort_index()
        logger.info(f"Combined data shape for {ticker}: {combined_df.shape}")
        logger.info(f"Combined data columns: {combined_df.columns.tolist()}")
        
        # Load and merge quote data
        quotes_df = process_quotes(ticker, data_loader.bucket, s3)
        if not quotes_df.empty:
            combined_df = pd.merge(
                combined_df,
                quotes_df[['bid_ask_spread', 'mid_price']],
                left_index=True,
                right_index=True,
                how='outer'
            ).ffill()
            logger.info(f"After quote merge shape: {combined_df.shape}")
            logger.info(f"After quote merge columns: {combined_df.columns.tolist()}")
        
        # Add missing columns with defaults
        if 'days_since_dividend' not in combined_df.columns:
            combined_df['days_since_dividend'] = 3650
            logger.info("Added default days_since_dividend")
        if 'split_ratio' not in combined_df.columns:
            combined_df['split_ratio'] = 1.0
            logger.info("Added default split_ratio")
            
        # Create sequences
        sequences = data_loader.create_sequences(combined_df)
        logger.info(f"Created sequences for {ticker} with shape {sequences.shape}")
        logger.info(f"Sequence features: {sequences.shape[-1]}")
        
        # Clear memory
        del dfs
        del combined_df
        del quotes_df
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
        logger.info(f"Data loader feature columns: {data_loader.feature_columns}")
        
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
                        logger.info(f"Sequences shape: {sequences.shape}")
                        logger.info(f"Sequences features: {sequences.shape[-1]}")
                except Exception as e:
                    logger.error(f"❌ Failed to load {ticker}: {str(e)}")
        
        if not all_sequences:
            raise ValueError("No data loaded for any tickers")
        
        # Combine all sequences
        combined_sequences = np.concatenate(all_sequences)
        logger.info(f"📊 Combined sequences shape: {combined_sequences.shape}")
        logger.info(f"Combined sequences features: {combined_sequences.shape[-1]}")
        
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