import os
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.py.ml_core.shap_optimizer import EnhancedSHAPOptimizer
from src.py.ml_core.data_loader import EnhancedDataLoader
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_ticker_data(ticker: str, data_loader: EnhancedDataLoader) -> np.ndarray:
    """Load and process data for a single ticker"""
    try:
        df = data_loader.load_ticker_data(ticker)
        if df is None:
            logger.warning(f"Failed to load data for {ticker}")
            return None
        sequences = data_loader.create_sequences(df)
        logger.info(f"Loaded {ticker} with shape {sequences.shape}")
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