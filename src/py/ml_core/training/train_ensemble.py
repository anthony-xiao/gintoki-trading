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

        # Log available models in S3 before loading
        logger.info("🔍 Checking available models in S3...")
        
        # List all models in the directory
        prefix = f"{model_factory.model_prefix}"
        response = model_factory.s3_client.list_objects_v2(
            Bucket=model_factory.bucket,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            logger.warning("No models found in S3 bucket")
        else:
            # Group models by type
            model_types = {}
            for obj in response['Contents']:
                key = obj['Key']
                if not key.endswith('.h5') and not key.endswith('.npz'):
                    continue
                    
                # Extract model type from path
                parts = key.split('/')
                if len(parts) >= 3:
                    model_type = parts[1]  # e.g., 'volatility', 'transformer', etc.
                    if model_type not in model_types:
                        model_types[model_type] = []
                    model_types[model_type].append(key)
            
            # Log available models and their latest versions
            logger.info("📦 Available models in S3:")
            latest_version = None
            for model_type, paths in model_types.items():
                if paths:
                    # Sort by timestamp in filename to get latest
                    latest_path = max(paths)
                    # Extract version from path (e.g., 'v1_20240325_105823' -> 'v1')
                    version = latest_path.split('/')[-1].split('_')[0]
                    if latest_version is None:
                        latest_version = version
                    logger.info(f"  - {model_type}:")
                    logger.info(f"    Latest version: s3://{model_factory.bucket}/{latest_path}")
                    logger.info(f"    Version tag: {version}")
                    logger.info(f"    Total versions: {len(paths)}")
                    if len(paths) > 1:
                        logger.info(f"    All versions:")
                        for path in sorted(paths, reverse=True)[:5]:  # Show top 5 most recent
                            logger.info(f"      - s3://{model_factory.bucket}/{path}")
                        if len(paths) > 5:
                            logger.info(f"      ... and {len(paths)-5} more versions")

            # Use the latest version found if no version was specified
            if args.model_version is None and latest_version is not None:
                logger.info(f"Using latest version found: {latest_version}")
                args.model_version = latest_version
            elif args.model_version is not None:
                logger.info(f"Using specified version: {args.model_version}")

        # Load pre-trained models and configurations
        logger.info("📦 Loading pre-trained models and configurations...")
        
        # Create model instances first
        model_factory.create_models()
        
        # Load pre-trained weights into the models
        model_factory.load_models(args.model_version)
        
        # Get the models from the factory
        volatility_model = model_factory.models['volatility'].model
        
        # Load the optimized transformer model specifically
        optimized_transformer = model_factory.load_model_from_s3('transformer_optimized', args.model_version)
        logger.info("✅ Loaded optimized transformer model")
        
        # Load feature mask and metadata
        feature_mask_path = f"models/features/models/test_v1_20250325_091600.npz"
        feature_mask_buffer = BytesIO(model_factory.s3_client.get_object(
            Bucket=model_factory.bucket,
            Key=feature_mask_path
        )['Body'].read())
        feature_mask_data = np.load(feature_mask_buffer, allow_pickle=True)
        
        feature_mask = feature_mask_data['mask']
        feature_metadata = feature_mask_data['metadata'].item()
        logger.info(f"✅ Loaded feature mask with {len(feature_mask)} features")

        # Train final ensemble using ModelFactory
        logger.info("🚢 Phase 6/6: Training adaptive ensemble...")
        ensemble = model_factory.ensemble
        
        # Preprocess data with feature mask and metadata
        processed_data = ensemble.preprocess_data(
            combined_data,
            feature_mask=feature_mask,  # Use the feature mask from SHAP optimization
            feature_metadata=feature_metadata  # Use the feature metadata from SHAP optimization
        )
        
        # Train XGBoost model
        logger.info("Training XGBoost momentum model...")
        
        # Log initial data state
        logger.info(f"Initial combined data shape: {combined_data.shape}")
        logger.info(f"Initial combined data columns: {combined_data.columns.tolist()}")
        
        # Get selected features from metadata and remove duplicates
        selected_features = list(dict.fromkeys(feature_metadata.get('selected_features', ensemble.feature_columns)))
        logger.info(f"Using {len(selected_features)} unique features for XGBoost: {selected_features}")
        
        # Ensure we have the required features for XGBoost
        if not all(feature in processed_data.columns for feature in selected_features):
            missing_features = [f for f in selected_features if f not in processed_data.columns]
            raise ValueError(f"Missing required features: {missing_features}")
            
        # Log data before feature selection
        logger.info(f"Data shape before feature selection: {processed_data.shape}")
        logger.info(f"Data columns before feature selection: {processed_data.columns.tolist()}")
        
        # Select features and verify data
        xgb_features = processed_data[selected_features].values
        if len(xgb_features) == 0:
            logger.error(f"Processed data shape: {processed_data.shape}")
            logger.error(f"Selected features: {selected_features}")
            logger.error(f"Available columns: {processed_data.columns.tolist()}")
            logger.error("Data sample:")
            logger.error(processed_data.head())
            raise ValueError("No valid data available for XGBoost training")
            
        # Create labels and ensure they match feature length
        xgb_labels = np.where(processed_data['close'].shift(-1) > processed_data['close'], 1, 0)[:-1]
        if len(xgb_labels) == 0:
            raise ValueError("No valid labels available for XGBoost training")
            
        # Remove the last row from features to match labels length
        xgb_features = xgb_features[:-1]
        
        # Log shapes for debugging
        logger.info(f"XGBoost training data shapes - Features: {xgb_features.shape}, Labels: {xgb_labels.shape}")
        
        # Verify data quality
        if np.any(np.isnan(xgb_features)):
            logger.error("NaN values found in features")
            raise ValueError("NaN values in features")
            
        if np.any(np.isinf(xgb_features)):
            logger.error("Infinite values found in features")
            raise ValueError("Infinite values in features")
        
        # Train the model
        logger.info("Starting XGBoost training...")
        ensemble.models['xgb_momentum'].fit(xgb_features, xgb_labels)
        
        # Log XGBoost training results
        logger.info("XGBoost training completed successfully")
        logger.info("XGBoost model details:")
        logger.info(f"  - Number of trees: {len(ensemble.models['xgb_momentum'].get_booster().get_dump())}")
        logger.info(f"  - Feature importance:")
        feature_importance = ensemble.models['xgb_momentum'].feature_importances_
        for feature, importance in zip(selected_features, feature_importance):
            logger.info(f"    - {feature}: {importance:.4f}")
        
        # Train Isolation Forest with required features
        logger.info("Training Isolation Forest...")
        # Ensure required features exist
        required_features = ['returns', 'volume_z', 'spread_ratio']
        for feature in required_features:
            if feature not in processed_data.columns:
                logger.warning(f"Required feature {feature} not found in processed data. Recalculating...")
                if feature == 'returns':
                    processed_data['returns'] = np.log(processed_data['close']/processed_data['close'].shift(1))
                elif feature == 'volume_z':
                    processed_data['volume_z'] = (processed_data['volume'] - processed_data['volume'].rolling(50).mean()) \
                                               / processed_data['volume'].rolling(50).std()
                elif feature == 'spread_ratio':
                    processed_data['spread_ratio'] = processed_data['bid_ask_spread'] / processed_data['mid_price']
        
        # Handle NaN values for Isolation Forest
        anomaly_features = processed_data[required_features].copy()
        
        # Log NaN values before handling
        nan_counts = anomaly_features.isna().sum()
        if nan_counts.any():
            logger.info("NaN values in anomaly features before handling:")
            for col, count in nan_counts.items():
                if count > 0:
                    logger.info(f"  {col}: {count} NaN values")
        
        # Handle NaN values
        for col in required_features:
            if anomaly_features[col].isna().any():
                # For derived features, use forward fill
                if col in ['returns', 'volume_z', 'spread_ratio']:
                    anomaly_features[col] = anomaly_features[col].ffill()
                else:
                    # For other features, use median
                    anomaly_features[col] = anomaly_features[col].fillna(anomaly_features[col].median())
        
        # Verify no NaN values remain
        if anomaly_features.isna().any().any():
            raise ValueError("NaN values still present in anomaly features after handling")
        
        if len(anomaly_features) == 0:
            raise ValueError("No valid data available for Isolation Forest training")
            
        # Log final data shape and statistics
        logger.info(f"Isolation Forest training data shape: {anomaly_features.shape}")
        logger.info("Feature statistics:")
        for col in required_features:
            stats = anomaly_features[col].describe()
            logger.info(f"  {col}:")
            logger.info(f"    - Mean: {stats['mean']:.4f}")
            logger.info(f"    - Std: {stats['std']:.4f}")
            logger.info(f"    - Min: {stats['min']:.4f}")
            logger.info(f"    - Max: {stats['max']:.4f}")
        
        # Train Isolation Forest
        ensemble.models['isolation_forest'].fit(anomaly_features.values)
        logger.info("Isolation Forest training completed successfully")
        
        # Calculate initial signals using optimized transformer
        logger.info("Calculating initial ensemble signals...")
        signals = ensemble.calculate_signals(
            processed_data,
            volatility_model=model_factory.models['volatility'].model,
            transformer_model=model_factory.models['transformer_optimized'].model  # Use optimized transformer
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
            'feature_metadata': feature_metadata  # Include feature metadata in the output
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