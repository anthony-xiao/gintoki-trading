import argparse
import pandas as pd
import numpy as np
import joblib
from src.py.ml_core.data_loader import EnhancedDataLoader
from src.py.ml_core.model_factory import ModelFactory
import logging
import time
from logging.handlers import RotatingFileHandler
import json
from io import BytesIO
import tensorflow as tf

def configure_logging():
    logger = logging.getLogger("ensemble_training")
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    
    file_handler = RotatingFileHandler(
        'ensemble_training.log', 
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

def load_pretrained_models(model_factory, model_version):
    """Load all necessary pre-trained models and configurations."""
    logger.info("📦 Loading pre-trained models and configurations...")
    
    try:
        # Load feature mask and metadata using ModelFactory's method
        feature_mask_path = model_factory._get_s3_key('features', model_version, extension='.npz')
        feature_mask_data = np.load(model_factory.s3_client.get_object(
            Bucket=model_factory.bucket,
            Key=feature_mask_path
        )['Body'].read())
        
        feature_mask = feature_mask_data['mask']
        feature_metadata = feature_mask_data['metadata'].item()
        logger.info(f"✅ Loaded feature mask with {len(feature_mask)} features")
    except Exception as e:
        logger.error(f"Failed to load feature mask: {str(e)}")
        raise

    try:
        # Create model instances first
        model_factory.create_models()
        
        # Load pre-trained weights into the models
        model_factory.load_models(model_version)
        
        # Get the models from the factory
        volatility_model = model_factory.models['volatility'].model
        
        # Load the optimized transformer model specifically
        optimized_transformer = model_factory.load_model_from_s3('transformer_optimized', model_version)
        logger.info("✅ Loaded optimized transformer model")
        
        logger.info("✅ Loaded all pre-trained models")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise

    return {
        'feature_mask': feature_mask,
        'feature_metadata': feature_metadata,
        'volatility_model': volatility_model,
        'transformer_model': optimized_transformer  # Use the optimized transformer
    }

def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Train ensemble model for trading')
        parser.add_argument('--tickers', nargs='+', required=True, help='List of tickers to train on')
        parser.add_argument('--model-version', default=None, help='Version tag for saved models')
        args = parser.parse_args()

        # Initialize data loader
        loader = EnhancedDataLoader()

        # Model configuration (reuse the same config from train.py)
        model_config = {
            'seq_length': 30,
            'd_model': 64,
            'num_heads': 8,
            'risk_management': {
                'max_vol': 0.015,
                'max_spread': 0.002
            }
        }

        model_factory = ModelFactory(model_config)

        # Load all data
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

        # Load pre-trained models and configurations
        pretrained_models = load_pretrained_models(model_factory, args.model_version)

        # Train ensemble
        logger.info("🚢 Training adaptive ensemble...")
        ensemble = model_factory.ensemble
        
        # Preprocess data with loaded feature mask
        processed_data = ensemble.preprocess_data(
            combined_data,
            feature_mask=pretrained_models['feature_mask'],
            feature_metadata=pretrained_models['feature_metadata']
        )
        
        # Train XGBoost model
        logger.info("Training XGBoost momentum model...")
        xgb_features = processed_data[ensemble.feature_columns].values
        xgb_labels = np.where(processed_data['close'].shift(-1) > processed_data['close'], 1, 0)[:-1]
        ensemble.models['xgb_momentum'].fit(xgb_features[:-1], xgb_labels)
        
        # Train Isolation Forest
        logger.info("Training Isolation Forest...")
        anomaly_features = processed_data[['returns', 'volume_z', 'spread_ratio']].values
        ensemble.models['isolation_forest'].fit(anomaly_features)
        
        # Calculate initial signals using loaded models
        logger.info("Calculating initial ensemble signals...")
        signals = ensemble.calculate_signals(
            processed_data,
            volatility_model=pretrained_models['volatility_model'],
            transformer_model=pretrained_models['transformer_model']  # Using optimized transformer
        )
        
        # Analyze and log signal performance
        logger.info("📊 Analyzing ensemble signal performance...")
        
        # Calculate signal statistics
        signal_values = signals['signal']
        signal_confidence = signals['confidence']
        regime_distribution = signals['regime']
        
        # Calculate basic metrics
        positive_signals = np.sum(signal_values > 0)
        negative_signals = np.sum(signal_values < 0)
        neutral_signals = np.sum(signal_values == 0)
        total_signals = len(signal_values)
        
        # Calculate regime distribution
        regime_counts = pd.Series(regime_distribution).value_counts()
        regime_percentages = regime_counts / total_signals * 100
        
        # Log performance metrics
        logger.info(f"Signal Distribution:")
        logger.info(f"  - Positive signals: {positive_signals} ({positive_signals/total_signals*100:.1f}%)")
        logger.info(f"  - Negative signals: {negative_signals} ({negative_signals/total_signals*100:.1f}%)")
        logger.info(f"  - Neutral signals: {neutral_signals} ({neutral_signals/total_signals*100:.1f}%)")
        
        logger.info(f"Regime Distribution:")
        for regime, count in regime_counts.items():
            logger.info(f"  - {regime}: {count} ({regime_percentages[regime]:.1f}%)")
        
        # Calculate confidence metrics
        avg_confidence = np.mean(signal_confidence)
        high_confidence_signals = np.sum(signal_confidence > 0.7)
        logger.info(f"Signal Confidence:")
        logger.info(f"  - Average confidence: {avg_confidence:.3f}")
        logger.info(f"  - High confidence signals (>0.7): {high_confidence_signals} ({high_confidence_signals/total_signals*100:.1f}%)")
        
        # Save signal analysis to metadata
        signal_analysis = {
            'signal_distribution': {
                'positive': int(positive_signals),
                'negative': int(negative_signals),
                'neutral': int(neutral_signals),
                'total': int(total_signals)
            },
            'regime_distribution': regime_counts.to_dict(),
            'confidence_metrics': {
                'average': float(avg_confidence),
                'high_confidence_count': int(high_confidence_signals)
            }
        }
        
        # Save all models and ensemble configuration to S3
        model_paths = model_factory.save_models(args.model_version)
        
        # Log all saved model paths
        logger.info("📝 Saved model paths:")
        for model_name, s3_path in model_paths.items():
            logger.info(f"  - {model_name}: s3://{model_factory.bucket}/{s3_path}")
            
        # Save model paths and signal analysis to a metadata file for future reference
        metadata = {
            'model_paths': model_paths,
            'version': args.model_version,
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'config': model_config,
            'signal_analysis': signal_analysis,
            'feature_metadata': pretrained_models['feature_metadata']
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

        logger.info("✅ Ensemble training completed successfully")

    except Exception as e:
        logger.error(f"🚨 Critical failure: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 