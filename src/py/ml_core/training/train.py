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

def main():
    try: 
        logger.info("🚀 Starting training pipeline")
        start_time = time.time()
        parser = argparse.ArgumentParser(description='Enhanced Training Pipeline')
        parser.add_argument('--tickers', nargs='+', default=['AMZN', 'TSLA', 'NVDA'])
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

        # 1. Train volatility detector
        logger.info("🔍 Phase 1/5: Training volatility detector...")
        detector = EnhancedVolatilityDetector(lookback=args.seq_length)
        logger.info(detector)
        detector.train(args.tickers, args.epochs)
        regime_model_model_path = 'src/py/ml_core/models/regime_model.h5'
        if not os.path.exists(regime_model_model_path):
            logger.error(f"❌ Model file not found: {os.path.abspath(regime_model_model_path)}")
            raise FileNotFoundError(f"Volatility model not generated at {regime_model_model_path}")
        registry.save_enhanced_model(regime_model_model_path, 'volatility')
        logger.info(f"✅ Volatility detector trained ({time.time()-start_time:.1f}s)")

        # 2. Prepare data for SHAP and Transformer
        logger.info("📦 Phase 2/5: Preparing training data...")
        data = pd.concat([loader.load_ticker_data(t) for t in args.tickers])
        assert set(FEATURE_COLUMNS).issubset(data.columns), \
            f"Missing features: {set(FEATURE_COLUMNS) - set(data.columns)}"
        # Create sequences and labels
        tf_dataset = loader.create_tf_dataset(data[FEATURE_COLUMNS], window=args.seq_length)
        X = np.array(list(tf_dataset.as_numpy_iterator()))
        y = np.where(data['close'].shift(-1) > data['close'], 1, -1)[args.seq_length:]

        # X = loader.create_sequences(data[FEATURE_COLUMNS], window=args.seq_length)
        # y = np.where(data['close'].shift(-1) > data['close'], 1, -1)[args.seq_length:]
        
        # Save for SHAP and Transformer
        joblib.dump(X, 'training_data.pkl')
        np.savez('transformer_data.npz', X=X, y=y)
        logger.debug(f"Data shape: {data.shape} | Columns: {data.columns.tolist()}")

        # 3. SHAP feature optimization
        logger.info("🎯 Phase 3/5: Running SHAP optimization...")
        optimizer = EnhancedSHAPOptimizer()
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
        
        # Initialize and process data through the ensemble
        ensemble = AdaptiveEnsembleTrader(ensemble_config)

        # 1. Preprocess the raw data using the ensemble's processor
        processed_data = ensemble.preprocess_data(data)

        # 2. Generate training signals from processed data
        signals = ensemble.calculate_signals(processed_data)

        # 3. Save the configured ensemble
        registry.save_enhanced_model('adaptive_ensemble.pkl', 'ensemble')


        print("\nTraining completed successfully!")

        # Add epoch progress logging
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logger.debug(
                    f"Transformer Epoch {epoch+1}/{args.epochs} | "
                    f"Loss: {logs['loss']:.4f} | Val Loss: {logs['val_loss']:.4f}"
                )
        
        transformer.train(
            'transformer_data.npz', 
            epochs=args.epochs,
            callbacks=[ProgressCallback()]
        )
        
        logger.info(f"🏁 Training completed in {time.time()-start_time:.1f} seconds")

    except Exception as e:
        logger.error(f"🚨 Critical failure: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
