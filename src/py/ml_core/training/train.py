import argparse
from volatility_regime import VolatilityRegimeDetector
from shap_optimizer import FeatureOptimizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--shap-samples', type=int, default=1000)
    args = parser.parse_args()
    
    # 1. Train volatility detector
    detector = VolatilityRegimeDetector()
    detector.train()
    
    # 2. Optimize features
    optimizer = FeatureOptimizer()
    top_features = optimizer.optimize_features('data/training.npz')
    print(f"Top features: {top_features}")
    
    # 3. Export artifacts
    detector.model.save('models/regime_detector.h5')
    np.savez('models/feature_mask.npz', mask=top_features)

if __name__ == "__main__":
    main()
