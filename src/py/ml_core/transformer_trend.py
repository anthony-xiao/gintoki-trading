import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
from src.py.ml_core.data_loader import EnhancedDataLoader
import logging
import os

class TransformerTrendAnalyzer:
    def __init__(self, seq_length=60, d_model=64, num_heads=8, feature_mask=None):
        self.seq_length = seq_length
        self.d_model = d_model
        self.feature_mask = feature_mask
        self.data_loader = EnhancedDataLoader()  # Initialize data_loader first
        
        # Log feature selection
        if feature_mask is not None:
            selected_features = [self.data_loader.feature_columns[i] for i in feature_mask]
            logging.info(f"Using optimized features: {selected_features}")
        
        # Build model after data_loader is initialized
        self.model = self._build_model(num_heads)
        
    def _build_model(self, num_heads):
        """Build transformer model with optional feature masking"""
        # Determine input shape based on feature mask
        if self.feature_mask is not None:
            input_shape = (self.seq_length, len(self.feature_mask))
        else:
            input_shape = (self.seq_length, len(self.data_loader.feature_columns))
            
        inputs = tf.keras.Input(shape=input_shape)
        
        # Transformer Encoder
        x = tf.keras.layers.Dense(self.d_model)(inputs)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = MultiHeadAttention(num_heads=num_heads, key_dim=self.d_model)(x, x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Trend Prediction
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='tanh')(x)  # [-1, 1] trend score
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )
        
        return model
    
    def train(self, data_path, epochs=50, batch_size=1024):
        """Train on preprocessed sequences with feature masking"""
        data = np.load(data_path)
        X, y = data['X'], data['y']
        
        # Apply feature mask if available
        if self.feature_mask is not None:
            X = X[:, :, self.feature_mask]
            logging.info(f"Applied feature mask, new shape: {X.shape}")
        
        # Convert labels to trend direction [-1, 1]
        y = np.where(y > 0, 1, -1).astype(np.float32)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
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
        logging.info("\U0001F389 Transformer training completed")
        
        # Save final weights with feature metadata
        model_path = 'src/py/ml_core/models/transformer_trend.h5'
        self.model.save(model_path)
        
        # Save feature metadata
        if self.feature_mask is not None:
            metadata = {
                'feature_mask': self.feature_mask.tolist(),
                'selected_features': [self.data_loader.feature_columns[i] for i in self.feature_mask],
                'seq_length': self.seq_length,
                'd_model': self.d_model,
                'num_heads': num_heads
            }
            np.savez(f"{model_path}.metadata.npz", **metadata)
        
        logging.info(f"Model saved to {os.path.abspath(model_path)}")

    def predict_trend_strength(self, sequences):
        """Predict normalized trend strength (-1 to 1) with feature masking"""
        if self.feature_mask is not None:
            sequences = sequences[:, :, self.feature_mask]
        return self.model.predict(sequences, verbose=0)
