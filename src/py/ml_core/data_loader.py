import boto3
import pandas as pd
import numpy as np
from io import BytesIO
from tqdm import tqdm
from typing import List, Dict

class EnhancedDataLoader:
    def __init__(self, bucket: str = 'quant-trader-data-gintoki'):
        self.s3 = boto3.client('s3')
        self.bucket = bucket
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume', 'vwap',
            'days_since_dividend', 'split_ratio', 'bid_ask_spread', 'mid_price'
        ]
        self.corporate_actions = self._load_corporate_actions()
    
    def _load_corporate_actions(self) -> Dict[str, pd.DataFrame]:
        """Load corporate actions from split S3 paths"""
        actions = []
        
        # Load dividends
        dividend_objs = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix='corporate_actions/dividends/'
        ).get('Contents', [])
        
        for obj in dividend_objs:
            if obj['Key'].endswith('.parquet'):
                response = self.s3.get_object(Bucket=self.bucket, Key=obj['Key'])
                df = pd.read_parquet(BytesIO(response['Body'].read()))
                df['type'] = 'dividend'  # Add type column
                actions.append(df)

        # Load splits
        split_objs = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix='corporate_actions/splits/'
        ).get('Contents', [])
        
        for obj in split_objs:
            if obj['Key'].endswith('.parquet'):
                response = self.s3.get_object(Bucket=self.bucket, Key=obj['Key'])
                df = pd.read_parquet(BytesIO(response['Body'].read()))
                df['type'] = 'split'  # Add type column
                actions.append(df)

        if not actions:
            return {}
        all_actions = pd.concat(actions)
        return {ticker: group for ticker, group in all_actions.groupby('symbol')}


    def load_ticker_data(self, ticker: str) -> pd.DataFrame:
        """Load and enhance data for a single ticker"""
        # Load core OHLCV
        ohlcv = self._load_s3_data(f'historical/{ticker}/aggregates/')
        
        # Merge corporate actions
        ohlcv = self._merge_corporate_actions(ohlcv, ticker)
        
        # Add quote features
        quotes = self._process_quotes(ticker)
        return pd.merge(ohlcv, quotes, left_index=True, right_index=True, how='left')

    def _load_s3_data(self, prefix: str) -> pd.DataFrame:
        """Load and concatenate parquet files from S3"""
        dfs = []
        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.parquet'):
                    response = self.s3.get_object(Bucket=self.bucket, Key=obj['Key'])
                    dfs.append(pd.read_parquet(BytesIO(response['Body'].read())))
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs).sort_index()

    def _merge_corporate_actions(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Enrich data with corporate action features"""
        ca = self.corporate_actions.get(ticker, pd.DataFrame())
        
        # Dividend features
        if not ca.empty and 'dividend' in ca['type'].values:
            dividends = ca[ca['type'] == 'dividend']
            df['days_since_dividend'] = df.index.map(
                lambda x: (x - dividends[dividends['ex_date'] < x]['ex_date'].max()).days
                if not dividends.empty else 3650
            ).fillna(3650)
        else:
            df['days_since_dividend'] = 3650

        # Split features
        if not ca.empty and 'split' in ca['type'].values:
            splits = ca[ca['type'] == 'split']
            df['split_ratio'] = df.index.map(
                lambda x: splits[splits['ex_date'] < x]['ratio'].iloc[-1]
                if not splits.empty else 1.0
            )
        else:
            df['split_ratio'] = 1.0
            
        return df

    def _process_quotes(self, ticker: str) -> pd.DataFrame:
        """Process raw quotes into spread features"""
        quotes = self._load_s3_data(f'historical/{ticker}/quotes/')
        # Handle empty case first
        if quotes.empty:
            return pd.DataFrame(columns=['bid_price', 'ask_price', 'bid_size', 'ask_size'], 
                            index=pd.DatetimeIndex([]))

        return quotes.resample('1min').agg({
            'bid_price': 'mean',
            'ask_price': 'mean',
            'bid_size': 'sum', 
            'ask_size': 'sum'
        }).assign(
            bid_ask_spread=lambda x: x['ask_price'] - x['bid_price'],
            mid_price=lambda x: (x['ask_price'] + x['bid_price']) / 2
        ).dropna()

    def create_sequences(self, data: pd.DataFrame, window: int = 60) -> np.ndarray:
        """Convert DataFrame to LSTM input sequences"""
        return np.array([
            data.iloc[i-window:i][self.feature_columns].values
            for i in range(window, len(data))
        ])
