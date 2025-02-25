import boto3
import pandas as pd
import numpy as np
import shap
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit

class TradingModelTrainer:
    def __init__(self, bucket='quant-trader-data-gintoki'):
        self.s3 = boto3.client('s3')
        self.bucket = bucket
        self.scaler = RobustScaler()
        
    def load_s3_data(self, prefix='processed/ticks/'):
        """Load and merge parquet files from S3"""
        objects = self.s3.list_objects(Bucket=self.bucket, Prefix=prefix)['Contents']
        dfs = []
        for obj in objects:
            response = self.s3.get_object(Bucket=self.bucket, Key=obj['Key'])
            dfs.append(pd.read_parquet(response['Body']))
        return pd.concat(dfs).sort_values('timestamp')

    def create_sequences(self, data, window=60):
        """Create LSTM-ready sequences"""
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i])
            y.append(data[i]['close'] > data[i-1]['close'])  # Binary classification
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """LSTM architecture from our initial research"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def shap_analysis(self, model, X_train):
        """Feature importance calculation"""
        explainer = shap.DeepExplainer(model, X_train[:100])
        shap_values = explainer.shap_values(X_train[:100])
        return shap_values

    def train(self):
        # Load and prepare data
        df = self.load_s3_data()
        features = df[['rsi', 'obv', 'vwap', 'volume']].dropna()
        scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = self.create_sequences(scaled)
        
        # Time-series split
        tscv = TimeSeriesSplit(n_splits=5)
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Build and train
            model = self.build_model((X_train.shape[1], X_train.shape[2]))
            model.fit(X_train, y_train, 
                     validation_data=(X_val, y_val),
                     epochs=20, batch_size=2048,
                     verbose=1 if fold ==0 else 0)  # Only show first fold
            
            # SHAP analysis
            if fold == 0:  # First fold only for efficiency
                shap_values = self.shap_analysis(model, X_train)
                pd.DataFrame(shap_values[0].mean(0), 
                            index=features.columns).to_csv('shap_values.csv')
            
            # Save best model
            if model.evaluate(X_val, y_val)[1] > best_score:
                model.save('best_model.h5')

if __name__ == "__main__":
    trainer = TradingModelTrainer()
    trainer.train()
