import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import shap

s3 = boto3.client('s3')

class TradingModel:
    def __init__(self):
        self.model = self.build_lstm()
        
    def build_lstm(self):
        model = Sequential([
            LSTM(128, input_shape=(60, 20)),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model
    
    def load_s3_data(self, bucket, key):
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(obj['Body'])
        return self.preprocess(df)
    
    def preprocess(self, df):
        # Add your feature engineering here
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y)
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val))
    
    def explain(self, X):
        explainer = shap.DeepExplainer(self.model, X)
        return explainer.shap_values(X)

if __name__ == '__main__':
    model = TradingModel()
    data = model.load_s3_data('s3://quant-trader-data-gintoki/historical', 'historical/AAPL/minute/20200101_to_20231231.parquet')
    model.train(data)
    model.model.save('model.h5')
