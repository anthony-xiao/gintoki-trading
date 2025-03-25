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

def list_available_models(s3_client, bucket, prefix):
    """List all available models in S3."""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' in response:
            models = [obj['Key'] for obj in response['Contents']]
            logger.info(f"Available models in s3://{bucket}/{prefix}:")
            for model in models:
                logger.info(f"  - {model}")
            return models
        else:
            logger.warning(f"No models found in s3://{bucket}/{prefix}")
            return []
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        return []

def load_pretrained_models(model_factory, model_version):
    """Load all necessary pre-trained models and configurations."""
    logger.info("📦 Loading pre-trained models and configurations...")
    
    # First, check what models are available
    available_models = list_available_models(
        model_factory.s3_client,
        model_factory.bucket,
        model_factory.model_prefix
    )
    
    if not available_models:
        raise ValueError("No pre-trained models found in S3. Please run train.py first to generate the models.")
    
    # Load feature mask and metadata
    feature_mask_path = f"{model_factory.model_prefix}features/enhanced_feature_mask.npz"
    if model_version:
        feature_mask_path = f"{feature_mask_path}_{model_version}"
    
    try:
        feature_mask_data = np.load(model_factory.s3_client.get_object(
            Bucket=model_factory.bucket,
            Key=feature_mask_path
        )['Body'].read())
        
        feature_mask = feature_mask_data['mask']
        feature_metadata = feature_mask_data['metadata'].item()
        logger.info(f"✅ Loaded feature mask with {len(feature_mask)} features")
    except Exception as e:
        logger.error(f"Failed to load feature mask: {str(e)}")
        logger.error("Please make sure you have run train.py first to generate the feature mask.")
        raise

    # Load volatility model
    volatility_path = f"{model_factory.model_prefix}volatility/model.h5"
    if model_version:
        volatility_path = f"{volatility_path}_{model_version}"
    
    try:
        volatility_model = tf.keras.models.load_model(model_factory.s3_client.get_object(
            Bucket=model_factory.bucket,
            Key=volatility_path
        )['Body'].read())
        logger.info("✅ Loaded volatility model")
    except Exception as e:
        logger.error(f"Failed to load volatility model: {str(e)}")
        logger.error("Please make sure you have run train.py first to generate the volatility model.")
        raise

    # Load optimized transformer model
    transformer_path = f"{model_factory.model_prefix}transformer_optimized/model.h5"
    if model_version:
        transformer_path = f"{transformer_path}_{model_version}"
    
    try:
        transformer_model = tf.keras.models.load_model(model_factory.s3_client.get_object(
            Bucket=model_factory.bucket,
            Key=transformer_path
        )['Body'].read())
        logger.info("✅ Loaded optimized transformer model")
    except Exception as e:
        logger.error(f"Failed to load transformer model: {str(e)}")
        logger.error("Please make sure you have run train.py first to generate the transformer model.")
        raise

    return {
        'feature_mask': feature_mask,
        'feature_metadata': feature_metadata,
        'volatility_model': volatility_model,
        'transformer_model': transformer_model
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
            transformer_model=pretrained_models['transformer_model']
        )
        
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
            'config': model_config,
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