import argparse
import numpy as np
import joblib
from ..data_loader import EnhancedDataLoader
from ..volatility_regime import EnhancedVolatilityDetector
from ..ensemble_strategy import EnhancedEnsembleTrader
from ..shap_optimizer import EnhancedSHAPOptimizer
from ..model_registry import EnhancedModelRegistry

def main():
    parser = argparse.ArgumentParser(description='Enhanced Training Pipeline')
    parser.add_argument('--tickers', nargs='+', default=['AMZN', 'TSLA', 'NVDA'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--shap-samples', type=int, default=2000)
    args = parser.parse_args()

    # Initialize components
    loader = EnhancedDataLoader()
    registry = EnhancedModelRegistry()
    
    # 1. Train volatility detector
    print("Training enhanced volatility detector...")
    detector = EnhancedVolatilityDetector()
    detector.train(args.tickers, args.epochs)
    registry.save_enhanced_model('regime_model.h5', 'volatility')
    
    # 2. SHAP feature optimization
    print("\nPerforming SHAP feature optimization...")
    data = pd.concat([loader.load_ticker_data(t) for t in args.tickers])
    X = loader.create_sequences(data)
    joblib.dump(X, 'training_data.pkl')
    
    optimizer = EnhancedSHAPOptimizer()
    top_features = optimizer.optimize_features('training_data.pkl')
    np.savez('enhanced_feature_mask.npz', mask=top_features)
    registry.save_enhanced_model('enhanced_feature_mask.npz', 'features')
    
    # 3. Train ensemble strategy
    print("\nTraining enhanced ensemble model...")
    ensemble = EnhancedEnsembleTrader()
    ensemble.train(args.tickers)
    joblib.dump(ensemble, 'enhanced_ensemble.pkl')
    registry.save_enhanced_model('enhanced_ensemble.pkl', 'ensemble')

if __name__ == "__main__":
    main()
