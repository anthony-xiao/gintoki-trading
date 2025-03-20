# src/py/ml_core/training/train.py
import argparse
import os
import pandas as pd
import numpy as np
import joblib
from src.py.ml_core.data_loader import EnhancedDataLoader
from src.py.ml_core.volatility_regime import EnhancedVolatilityDetector
from src.py.ml_core.ensemble_strategy import AdaptiveEnsembleTrader
from src.py.ml_core.shap_optimizer import EnhancedSHAPOptimizer
from src.py.ml_core.transformer_trend import TransformerTrendAnalyzer
from src.py.ml_core.model_registry import EnhancedModelRegistry
import logging
import time
from logging.handlers import RotatingFileHandler
import gc
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Configure logging at the top
def configure_logging():
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    
    # Console handler with progress formatting
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    
    # File handler with detailed logging
    file_handler = RotatingFileHandler(
        'training.log', 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s"
    )
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = configure_logging()

# Configure feature columns exactly as requested
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume', 'vwap',
    'days_since_dividend', 'split_ratio', 'bid_ask_spread', 'mid_price'
]

def load_ticker_data(ticker: str, loader: EnhancedDataLoader) -> pd.DataFrame:
    """Load data for a single ticker"""
    try:
        data = loader.load_ticker_data(ticker)
        if data is not None:
            logger.info(f"✅ Loaded {len(data)} rows for {ticker}")
            return data
        return None
    except Exception as e:
        logger.error(f"❌ Failed to load data for {ticker}: {str(e)}")
        return None

def process_chunk(chunk: pd.DataFrame, window_size: int, feature_columns: list) -> np.ndarray:
    """Process a chunk of data into sequences"""
    sequences = []
    for i in range(window_size, len(chunk)):
        seq = chunk.iloc[i-window_size:i][feature_columns].values
        if seq.shape == (window_size, len(feature_columns)):
            sequences.append(seq)
    return np.array(sequences, dtype=np.float32)

def main():
    try: 
        logger.info("🚀 Starting training pipeline")
        start_time = time.time()
        parser = argparse.ArgumentParser(description='Enhanced Training Pipeline')
        parser.add_argument('--tickers', nargs='+', default=['SCMI'])
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--shap-samples', type=int, default=2000)
        parser.add_argument('--seq-length', type=int, default=60,
                        help='Transformer sequence length (default: 60)')
        parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level (default: INFO)')
        args = parser.parse_args()

        # Initialize components
        loader = EnhancedDataLoader()
        registry = EnhancedModelRegistry()

        # Load all data in parallel
        logger.info("📦 Loading data for all tickers...")
        with ThreadPoolExecutor(max_workers=min(len(args.tickers), 4)) as executor:
            load_func = partial(load_ticker_data, loader=loader)
            all_data = list(filter(None, executor.map(load_func, args.tickers)))
        
        if not all_data:
            raise ValueError("🛑 No data loaded for any tickers")
        
        # Combine all ticker data
        combined_data = pd.concat(all_data)
        logger.info(f"📊 Total combined data shape: {combined_data.shape}")

        # 1. Train volatility detector
        logger.info("🔍 Phase 1/5: Training volatility detector...")
        detector = EnhancedVolatilityDetector(lookback=args.seq_length)
        logger.info(detector)
        detector.train(combined_data, args.epochs)
        
        # Save and register the model
        local_model_path = 'src/py/ml_core/models/regime_model.h5'
        if not os.path.exists(local_model_path):
            logger.error(f"❌ Model file not found: {os.path.abspath(local_model_path)}")
            raise FileNotFoundError(f"Volatility model not generated at {local_model_path}")
        
        s3_key = registry.save_enhanced_model(local_model_path, 'volatility')
        logger.info(f"✅ Model saved to S3: {s3_key}")
        
        logger.info(f"✅ Volatility detector trained ({time.time()-start_time:.1f}s)")

        # 2. Prepare data for SHAP and Transformer
        logger.info("📦 Phase 2/5: Preparing training data...")
        
        if combined_data.empty:
            raise ValueError("🛑 No training data available")
        assert set(FEATURE_COLUMNS).issubset(combined_data.columns), \
            f"Missing features: {set(FEATURE_COLUMNS) - set(combined_data.columns)}"
        
        # Process data in parallel chunks
        chunk_size = 500000  # Reduced from 1M for better memory management
        all_sequences = []
        total_rows = len(combined_data)
        
        logger.info(f"🔄 Processing {total_rows} rows in chunks of {chunk_size}")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            process_func = partial(process_chunk, window_size=args.seq_length, feature_columns=FEATURE_COLUMNS)
            futures = []
            
            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                chunk = combined_data.iloc[start_idx:end_idx]
                futures.append(executor.submit(process_func, chunk))
            
            for future in futures:
                chunk_array = future.result()
                if chunk_array.size > 0:
                    all_sequences.append(chunk_array)
                    logger.info(f"✅ Processed chunk, shape: {chunk_array.shape}")
                
                # Clear memory after each chunk
                gc.collect()
        
        # Combine all sequences
        X = np.concatenate(all_sequences, axis=0)
        logger.info(f"📊 Final sequence shape: {X.shape}")

        # Align labels with sequences
        y = np.where(combined_data['close'].shift(-1) > combined_data['close'], 1, -1)[args.seq_length:]

        # Save for SHAP and Transformer
        logger.info("💾 Saving processed data...")
        joblib.dump(X, 'training_data.pkl')
        np.savez('transformer_data.npz', X=X, y=y)
        logger.debug(f"Data shape: {combined_data.shape} | Columns: {combined_data.columns.tolist()}")

        # 3. SHAP feature optimization
        logger.info("🎯 Phase 3/5: Running SHAP optimization...")
        optimizer = EnhancedSHAPOptimizer(background_data=combined_data)
        top_features = optimizer.optimize_features('transformer_data.npz', top_k=15)
        np.savez('enhanced_feature_mask.npz', mask=top_features)
        registry.save_enhanced_model('enhanced_feature_mask.npz', 'features')
        logger.debug(f"Top features: {top_features}")

        # 4. Train Transformer Trend Analyzer
        logger.info("🧠 Phase 4/5: Training Transformer trend model...")
        transformer = TransformerTrendAnalyzer(seq_length=args.seq_length)
        transformer.train('transformer_data.npz', epochs=args.epochs)

        transformer_trend_model_path = 'src/py/ml_core/models/transformer_trend.h5'
        if not os.path.exists(transformer_trend_model_path):
            logger.error(f"❌ Model file not found: {os.path.abspath(transformer_trend_model_path)}")
            raise FileNotFoundError(f"Volatility model not generated at {transformer_trend_model_path}")
        registry.save_enhanced_model(transformer_trend_model_path, 'transformer')

        # 5. Train final ensemble
        logger.info("🚢 Phase 5/5: Training adaptive ensemble...")
        ensemble_config = {
            'feature_columns': FEATURE_COLUMNS,
            'volatility_model_path': 'src/py/ml_core/models/regime_model.h5',
            'transformer_model_path': 'src/py/ml_core/models/transformer_trend.h5',
            'risk_management': {
                'max_vol': 0.015,
                'max_spread': 0.002
            },
            'regime_weights': {
                'high_volatility': {'transformer': 0.6, 'xgb': 0.3, 'lstm': 0.1},
                'low_volatility': {'transformer': 0.3, 'xgb': 0.5, 'lstm': 0.2},
                'neutral': {'transformer': 0.4, 'xgb': 0.4, 'lstm': 0.2}
            }
        }
        
        ensemble = AdaptiveEnsembleTrader(ensemble_config)
        processed_data = ensemble.preprocess_data(combined_data)
        signals = ensemble.calculate_signals(processed_data)
        registry.save_enhanced_model('adaptive_ensemble.pkl', 'ensemble')

        logger.info(f"🏁 Training completed in {time.time()-start_time:.1f} seconds")

    except Exception as e:
        logger.error(f"🚨 Critical failure: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
