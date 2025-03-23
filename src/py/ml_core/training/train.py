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
from src.py.ml_core.model_factory import ModelFactory
import logging
import time
from logging.handlers import RotatingFileHandler
import gc
from typing import Dict
import json
from io import BytesIO
import tensorflow as tf

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

# Initialize logger at module level
logger = configure_logging()

# Configure feature columns exactly as requested
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume', 'vwap',
    'bid_ask_spread', 'days_since_dividend', 'split_ratio', 'mid_price'
]

def main():
    try:
        # Configure GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU memory growth enabled for {len(gpus)} devices")
            except RuntimeError as e:
                logger.error(f"Error configuring GPU: {str(e)}")
        
        # Set mixed precision training for better GPU performance
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Train ML models for trading')
        parser.add_argument('--tickers', nargs='+', required=True, help='List of tickers to train on')
        parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
        parser.add_argument('--shap-samples', type=int, default=1000, help='Number of SHAP samples')
        parser.add_argument('--seq-length', type=int, default=30, help='Sequence length for models')
        parser.add_argument('--log-level', default='INFO', help='Logging level')
        parser.add_argument('--model-version', default=None, help='Version tag for saved models')
        args = parser.parse_args()

        # Set up logging level from command line
        logger.setLevel(getattr(logging, args.log_level))

        # Initialize data loader
        loader = EnhancedDataLoader()

        # Model configuration
        model_config = {
            'seq_length': args.seq_length,  # Use command line argument
            'd_model': 64,
            'num_heads': 8,
            'risk_management': {
                'max_vol': 0.015,
                'max_spread': 0.002
            }
        }

        model_factory = ModelFactory(model_config)

        # Load all data once
        logger.info("📦 Loading data for all tickers...")
        all_data = {}
        for ticker in args.tickers:
            data = loader.load_ticker_data(ticker)
            if data is not None:
                all_data[ticker] = data
                logger.info(f"✅ Loaded {len(data)} rows for {ticker}")
        
        if not all_data:
            raise ValueError("🛑 No data loaded for any tickers")
        
        # Combine all ticker data
        combined_data = pd.concat(all_data.values())
        logger.info(f"📊 Total combined data shape: {combined_data.shape}")

        # 1. Train volatility detector using ModelFactory
        logger.info("🔍 Phase 1/6: Training volatility detector...")
        model_factory.create_models()  # Create model instances
        volatility_model = model_factory.models['volatility']
        volatility_model.train(combined_data, args.epochs)
        
        # Save volatility model to S3
        model_factory.save_model_to_s3(volatility_model.model, 'volatility', args.model_version)
        logger.info("✅ Volatility model saved to S3")

        # 2. Prepare data for Transformer and SHAP
        logger.info("📦 Phase 2/6: Preparing training data...")
        
        # Validate data exists and has required columns
        if combined_data.empty:
            raise ValueError("🛑 No training data available")
        assert set(FEATURE_COLUMNS).issubset(combined_data.columns), \
            f"Missing features: {set(FEATURE_COLUMNS) - set(combined_data.columns)}"
        
        # Create 3D sequences [samples, window, features] with GPU acceleration
        window_size = args.seq_length
        num_features = len(FEATURE_COLUMNS)
        
        # Convert data to GPU tensor for faster processing
        feature_data = combined_data[FEATURE_COLUMNS].values.astype(np.float32)
        feature_tensor = tf.convert_to_tensor(feature_data, dtype=tf.float32)
        
        # Process data in chunks with GPU acceleration
        chunk_size = 1000000  # Process 1M rows at a time
        all_sequences = []
        total_rows = len(combined_data)
        
        logger.info(f"🔄 Processing {total_rows} rows in chunks of {chunk_size}")
        
        # Create a TensorFlow dataset for efficient processing
        dataset = tf.data.Dataset.from_tensor_slices(feature_tensor)
        
        # Define sequence creation function
        def create_sequences(chunk):
            sequences = []
            for i in range(window_size, len(chunk)):
                seq = chunk[i-window_size:i]
                if seq.shape == (window_size, num_features):
                    sequences.append(seq)
            return np.array(sequences, dtype=np.float32)
        
        # Process chunks in parallel using tf.data
        dataset = dataset.batch(chunk_size)
        dataset = dataset.map(lambda x: tf.py_function(create_sequences, [x], tf.float32))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Process all chunks
        for chunk_sequences in dataset:
            if chunk_sequences.shape[0] > 0:  # Only process non-empty chunks
                all_sequences.append(chunk_sequences)
                logger.info(f"✅ Processed chunk, shape: {chunk_sequences.shape}")
            
            # Clear memory
            tf.keras.backend.clear_session()
            gc.collect()
        
        # Combine all sequences
        X = np.concatenate(all_sequences, axis=0)
        logger.info(f"📊 Final sequence shape: {X.shape}")

        # Align labels with sequences using GPU
        y = tf.where(
            tf.gather(combined_data['close'].values[1:], tf.range(len(combined_data)-1)) > 
            tf.gather(combined_data['close'].values[:-1], tf.range(len(combined_data)-1)),
            1, -1
        ).numpy()[window_size:]
        
        # Convert sequences to GPU tensors for faster processing
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.int32)
        
        # Create TensorFlow datasets with optimized pipeline
        train_ds = tf.data.Dataset.from_tensor_slices((X, y)) \
            .cache() \
            .shuffle(buffer_size=10000) \
            .batch(4096) \
            .prefetch(tf.data.AUTOTUNE)
        
        # Save for Transformer and SHAP
        logger.info("💾 Saving processed data...")
        joblib.dump(X.numpy(), 'training_data.pkl')
        np.savez('transformer_data.npz', X=X.numpy(), y=y.numpy())
        logger.debug(f"Data shape: {combined_data.shape} | Columns: {combined_data.columns.tolist()}")

        # 3. Train initial Transformer using ModelFactory
        logger.info("🧠 Phase 3/6: Training initial Transformer trend model...")
        transformer_model = model_factory.models['transformer']
        
        # Use the optimized train_ds for training
        transformer_model.model.fit(
            train_ds,
            epochs=args.epochs,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    'transformer_trend.h5',
                    save_best_only=True,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.EarlyStopping(
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        logger.info("✅ Initial transformer model training completed")

        # Save initial transformer model to S3
        model_factory.save_model_to_s3(transformer_model.model, 'transformer', args.model_version)
        logger.info("✅ Initial transformer model saved to S3")

        # 4. SHAP feature optimization
        logger.info("🎯 Phase 4/6: Running SHAP optimization...")
        optimizer = EnhancedSHAPOptimizer(
            model_path=None,  # Don't load from S3
            background_data=combined_data,
            background_samples=args.shap_samples,
            ticker=args.tickers[0],  # Use the first ticker for SHAP optimization
            data_loader=loader,  # Pass the existing data loader instance
            trained_volatility_model=volatility_model.model,  # Pass the volatility model we just trained
            trained_transformer_model=transformer_model.model  # Pass the transformer model we just trained
        )
        
        # Get optimized features
        top_features = optimizer.optimize_features('transformer_data.npz', top_k=15)
        
        # Save feature mask and metadata
        feature_metadata = {
            'selected_features': [FEATURE_COLUMNS[i] for i in top_features],
            'essential_features': optimizer.essential_features,
            'feature_weights': optimizer._trading_feature_weights().tolist(),
            'timestamp': time.strftime('%Y%m%d_%H%M%S')
        }
        
        np.savez('enhanced_feature_mask.npz', 
                 mask=top_features,
                 metadata=feature_metadata)
        
        # Save to S3 with versioning
        registry = EnhancedModelRegistry()
        registry.save_enhanced_model('enhanced_feature_mask.npz', 'features')
        logger.info(f"✅ Selected features: {feature_metadata['selected_features']}")

        # 5. Retrain Transformer with optimized features using ModelFactory
        logger.info("🧠 Phase 5/6: Retraining Transformer with optimized features...")
        optimized_transformer = model_factory.models['transformer']
        
        # Use the optimized train_ds for training
        optimized_transformer.model.fit(
            train_ds,
            epochs=args.epochs,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    'transformer_trend_optimized.h5',
                    save_best_only=True,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.EarlyStopping(
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        logger.info("✅ Optimized transformer model training completed")

        # Save optimized transformer model to S3
        model_factory.save_model_to_s3(optimized_transformer.model, 'transformer_optimized', args.model_version)
        logger.info("✅ Optimized transformer model saved to S3")

        # 6. Train final ensemble using ModelFactory
        logger.info("🚢 Phase 6/6: Training adaptive ensemble...")
        ensemble = model_factory.ensemble
        processed_data = ensemble.preprocess_data(combined_data)
        signals = ensemble.calculate_signals(processed_data)
        
        # Save all models and ensemble configuration to S3
        model_paths = model_factory.save_models(args.model_version)
        
        # Log all saved model paths
        logger.info("📝 Saved model paths:")
        for model_name, s3_path in model_paths.items():
            logger.info(f"  - {model_name}: s3://{model_factory.bucket}/{s3_path}")
            
        # Save model paths to a metadata file for future reference
        metadata = {
            'model_paths': model_paths,
            'version': args.model_version,
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'config': model_config
        }
        
        # Save metadata to S3
        metadata_buffer = BytesIO()
        metadata_buffer.write(json.dumps(metadata, indent=2).encode())
        metadata_buffer.seek(0)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        if args.model_version:
            metadata_key = f"{model_factory.model_prefix}metadata/models/{args.model_version}_{timestamp}.json"
        else:
            metadata_key = f"{model_factory.model_prefix}metadata/models/{timestamp}.json"
            
        model_factory.s3_client.upload_fileobj(
            metadata_buffer,
            model_factory.bucket,
            metadata_key,
            ExtraArgs={'ContentType': 'application/json'}
        )
        logger.info(f"✅ Training metadata saved to s3://{model_factory.bucket}/{metadata_key}")

        logger.info("✅ All models and ensemble configuration saved to S3")
        logger.info(f"🏁 Training completed in {time.time()-start_time:.1f} seconds")

    except Exception as e:
        logger.error(f"🚨 Critical failure: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
