from typing import List
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
from src.py.ml_core.data_loader import EnhancedDataLoader

class EnhancedVolatilityDetector:
    def __init__(self, lookback: int = 60, data_loader: EnhancedDataLoader = None):
        self.data_loader = data_loader if data_loader is not None else EnhancedDataLoader()
        self.model = self._build_model(lookback)
        self.lookback = lookback

    def _build_model(self, lookback: int) -> tf.keras.Model:
        """Construct LSTM architecture with attention"""
        inputs = tf.keras.Input(shape=(lookback, len(self.data_loader.feature_columns)))
        x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(query=x, value=x)
        x = tf.keras.layers.LSTM(64)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _calculate_event_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhance volatility calculation with corporate actions"""
        df = df.copy()
    
        # Handle zero/NaN in close prices
        df['close'] = df['close'].replace(0, np.nan).ffill()
        
        # Calculate returns safely
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        # Calculate volatility using price ranges
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['base_volatility'] = df['price_range'].rolling(20, min_periods=1).mean()
        # Calculate boosts with NaN protection
        df['div_boost'] = np.where(df['days_since_dividend'] < 5, 1.3, 1.0)
        df['split_boost'] = np.where(np.abs(df['split_ratio'] - 1.0) > 1e-8, 1.5 / (df['split_ratio'] + 1e-8), 1.0)
        df['event_volatility'] = df['base_volatility'] * df[['div_boost', 'split_boost']].max(axis=1)
        
        return df

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate regime labels with enhanced criteria"""
        df = self._calculate_event_volatility(df)
        
        # Regime conditions
        high_vol = (df['event_volatility'] > 0.015) 
        low_vol = (df['event_volatility'] < 0.005)
        high_spread = (df['bid_ask_spread'] > df['bid_ask_spread'].rolling(20).mean() * 1.5)
        
        df['regime'] = np.select(
            [high_vol & ~high_spread, low_vol & ~high_spread, high_spread],
            [2, 0, 1],  # 2=Momentum, 0=Mean Revert, 1=High Spread
            default=1
        )
        return df.dropna()

    def train_model(self, tickers: List[str] = None, epochs: int = 100):
        """Complete training implementation with data pipeline"""
        if not tickers:
            tickers = self.data_loader.get_available_tickers()
        
        # Generate training data through data loader
        data = pd.concat([self._process_ticker(t) for t in tickers])
        X = self.data_loader.create_sequences(data)
        y = data['regime'].values[self.lookback:]
        
        # Train with checkpoints
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=8192,
            validation_split=0.2,
            callbacks=[
                ModelCheckpoint('src/py/ml_core/models/regime_model.h5', 
                            save_best_only=True,
                            monitor='val_accuracy'),
                EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )
        # Save final weights
        self.model.save('src/py/ml_core/models/regime_model.h5')

    def _process_ticker(self, ticker: str) -> pd.DataFrame:
        """Process individual ticker data"""
        df = self.data_loader.load_ticker_data(ticker)
        return self.create_labels(df)

    def evaluate(self, ticker: str):
        """Enhanced evaluation with spread analysis"""
        df = self.data_loader.load_ticker_data(ticker)
        X = self.data_loader.create_sequences(df)
        y = df['regime'].values[self.lookback:]
        
        y_pred = np.argmax(self.model.predict(X), axis=1)
        print(classification_report(y, y_pred))
        
        # Spread impact analysis
        spread_impact = pd.DataFrame({
            'true_regime': y,
            'predicted_regime': y_pred,
            'spread': df['bid_ask_spread'].values[self.lookback:]
        })
        print("\nSpread Statistics by Regime:")
        print(spread_impact.groupby('predicted_regime')['spread'].describe())
