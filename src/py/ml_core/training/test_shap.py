import os
import numpy as np
import logging
from src.py.ml_core.shap_optimizer import EnhancedSHAPOptimizer
from src.py.ml_core.data_loader import EnhancedDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_shap_optimization():
    try:
        logger.info("🚀 Starting SHAP optimization test")
        
        # Initialize components
        data_loader = EnhancedDataLoader()
        optimizer = EnhancedSHAPOptimizer(background_samples=100)
        
        # Load some sample data
        logger.info("📦 Loading sample data...")
        df = data_loader.load_ticker_data('AMZN')
        if df is None:
            raise ValueError("Failed to load sample data")
            
        # Create sequences
        sequences = data_loader.create_sequences(df)
        logger.info(f"Created sequences shape: {sequences.shape}")
        
        # Save sequences for testing
        np.savez('test_data.npz', X=sequences)
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