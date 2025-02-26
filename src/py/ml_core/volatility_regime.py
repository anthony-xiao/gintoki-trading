import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report

class VolatilityRegimeDetector:
    def __init__(self, lookback=60, features=4):
        self.model = self.build_model(lookback, features)
        self.lookback = lookback
        self.scaler = RobustScaler()
    
    def build_model(self, lookback, features):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, 
                                input_shape=(lookback, features)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adamax',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def create_sequences(self, data):
        X, y = [], []
        for i in range(self.lookback, len(data)-1):
            X.append(data[i-self.lookback:i])
            y.append(data.iloc[i+1]['regime'])
        return np.array(X), np.array(y)
    
    def train(self, s3_path='s3://quant-trader-data-gintoki/processed/'):
        # Load and preprocess data
        df = self.load_s3_data(s3_path)
        df = self.label_regimes(df)
        
        # Create sequences
        scaled = self.scaler.fit_transform(df[['rsi', 'obv', 'vwap', 'volume']])
        X, y = self.create_sequences(pd.DataFrame(scaled, index=df.index))
        
        # Train/val split
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # Train with distributed strategy
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.model.fit(X_train, y_train,
                          validation_data=(X_val, y_val),
                          batch_size=8192,
                          epochs=100,
                          callbacks=[
                              ModelCheckpoint('regime_model.h5', save_best_only=True),
                              EarlyStopping(patience=5)
                          ])
        
        # Evaluate
        y_pred = np.argmax(self.model.predict(X_val), axis=1)
        print(classification_report(y_val, y_pred))
    
    def label_regimes(self, df):
        # Implement volatility regime labeling logic
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        df['regime'] = np.where(df['volatility'] > 0.015, 2,  # High vol
                               np.where(df['volatility'] < 0.005, 0, 1))  # Low/med
        return df.dropna()
