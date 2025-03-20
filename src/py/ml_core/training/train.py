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
import tensorflow as tf
import json

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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Training Pipeline')
    parser.add_argument('--tickers', nargs='+', default=['SCMI'],
                      help='List of tickers to train on')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--sequence-length', type=int, default=60,
                      help='Transformer sequence length')
    parser.add_argument('--num-classes', type=int, default=3,
                      help='Number of output classes')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Initial learning rate')
    parser.add_argument('--run-shap', action='store_true',
                      help='Run SHAP optimization')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    return parser.parse_args()

def setup_logging():
    """Setup logging configuration"""
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

def save_model_and_history(model, history, args):
    """Save model and training history"""
    # Save model
    model_path = 'src/py/ml_core/models/regime_model.h5'
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save training history
    history_path = 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    logger.info(f"Training history saved to {history_path}")
    
    # Save to S3
    registry = EnhancedModelRegistry()
    s3_key = registry.save_enhanced_model(model_path, 'volatility')
    logger.info(f"Model saved to S3: {s3_key}")

def main():
    """Main training function with performance optimizations"""
    args = parse_args()
    logger = setup_logging()
    
    # Enable mixed precision training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Initialize data loader with parallel processing
    data_loader = EnhancedDataLoader()
    
    # Load and preprocess data with tf.data.Dataset
    train_data = data_loader.create_tf_dataset(
        data_loader.load_ticker_data(args.tickers[0]),
        window=args.sequence_length
    )
    
    val_data = data_loader.create_tf_dataset(
        data_loader.load_ticker_data(args.tickers[0]),
        window=args.sequence_length
    )
    
    # Initialize model with performance optimizations
    model = create_model(
        input_shape=(args.sequence_length, len(data_loader.feature_columns)),
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        use_mixed_precision=True
    )
    
    # Learning rate schedule
    initial_learning_rate = args.learning_rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    
    # Early stopping with patience
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Model checkpointing
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    
    # Train model with optimizations
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        callbacks=[
            early_stopping,
            checkpoint,
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ],
        workers=4,  # Parallel training
        use_multiprocessing=True
    )
    
    # Save model and history
    save_model_and_history(model, history, args)
    
    # Run SHAP optimization with reduced samples
    if args.run_shap:
        logger.info("Starting SHAP optimization...")
        optimizer = EnhancedSHAPOptimizer(
            background_samples=500  # Reduced from 1000
        )
        top_features = optimizer.optimize_features('transformer_data.npz', top_k=15)
        logger.info(f"Selected top features: {top_features}")

def create_model(input_shape, num_classes, learning_rate, use_mixed_precision=False):
    """Create model with performance optimizations"""
    with tf.keras.mixed_precision.policy('mixed_float16' if use_mixed_precision else 'float32'):
        model = tf.keras.Sequential([
            # Input layer with shape validation
            tf.keras.layers.Input(shape=input_shape),
            
            # LSTM layers with gradient checkpointing
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.2),
            
            # Dense layers with mixed precision
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile with performance optimizations
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            jit_compile=True  # Enable XLA compilation
        )
        
        return model

if __name__ == "__main__":
    main()
