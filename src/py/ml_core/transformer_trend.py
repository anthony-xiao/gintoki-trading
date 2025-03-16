import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
from src.py.ml_core.data_loader import EnhancedDataLoader
import logging
import os

class TransformerTrendAnalyzer:
    def __init__(self, seq_length=60, d_model=64, num_heads=8):
        self.seq_length = seq_length
        self.d_model = d_model
        self.model = self._build_model(num_heads)
        self.data_loader = EnhancedDataLoader()
        
    def _build_model(self, num_heads):
        inputs = tf.keras.Input(shape=(self.seq_length, len(self.data_loader.feature_columns)))
        
        # Transformer Encoder
        x = tf.keras.layers.Dense(self.d_model)(inputs)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = MultiHeadAttention(num_heads=num_heads, key_dim=self.d_model)(x, x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Trend Prediction
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='tanh')(x)  # [-1, 1] trend score
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def train(self, data_path, epochs=50, batch_size=1024):
        """Train on preprocessed sequences"""
        data = np.load(data_path)
        X, y = data['X'], data['y']
        
        # Convert labels to trend direction [-1, 1]
        y = np.where(y > 0, 1, -1).astype(np.float32)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    'transformer_trend.h5',
                    save_best_only=True
                )
            ]
        )
        logging.info("\U0001F389 Volatility training completed")
        # Save final weights
        model_path = 'src/py/ml_core/models/transformer_trend.h5'  # Save in current directory
        self.model.save(model_path)
        logging.info(f"Model saved to {os.path.abspath(model_path)}")

    def predict_trend_strength(self, sequences):
        """Predict normalized trend strength (-1 to 1)"""
        return self.model.predict(sequences, verbose=0)
