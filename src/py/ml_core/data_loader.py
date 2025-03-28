import gc
import logging
import os
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import pandas as pd
import numpy as np
from io import BytesIO
from tqdm import tqdm
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf


load_dotenv()

class EnhancedDataLoader:
    def __init__(self, bucket: str = 'quant-trader-data-gintoki', feature_mask=None):
        logger = logging.getLogger("training")
        logger.debug("\U0001F310 Initializing S3 client for bucket %s", bucket)
        
        # Configure GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU memory growth enabled for {len(gpus)} devices")
            except RuntimeError as e:
                logger.error(f"Error configuring GPU: {str(e)}")
        
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-west-2')
        )
        self.bucket = bucket
        
        # Define base features (from raw data)
        self.base_feature_columns = [
            'open', 'high', 'low', 'close', 'volume', 'vwap',
            'bid_ask_spread', 'days_since_dividend', 'split_ratio', 'mid_price'
        ]
        
        # Define technical indicators
        self.technical_indicators = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'obv', 'adx', 'di_plus', 'di_minus'
        ]
        
        # Combine all possible features
        self.all_feature_columns = self.base_feature_columns + self.technical_indicators
        
        # Set feature columns based on mask
        self.feature_mask = feature_mask
        if feature_mask is not None:
            self.feature_columns = [self.all_feature_columns[i] for i in feature_mask]
            logger.info(f"Using optimized features: {self.feature_columns}")
        else:
            self.feature_columns = self.base_feature_columns  # Start with base features only
            logger.info("Using base features only")
        
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

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataset"""
        logger = logging.getLogger("training")  # Initialize logger
        try:
            # Ensure we have enough data for calculations
            if len(df) < 26:  # Minimum required for MACD
                logger.warning(f"Insufficient data for technical indicators: {len(df)} rows")
                return df
            
            # Use mid_price as close price if available, otherwise use close
            price_col = 'mid_price' if 'mid_price' in df.columns else 'close'
            
            # Ensure prices are valid
            if df[price_col].isna().any() or (df[price_col] == 0).any():
                logger.warning(f"Invalid {price_col} prices detected, filling with forward/backward fill")
                df[price_col] = df[price_col].replace(0, np.nan).ffill().bfill()
            
            # Calculate returns with proper handling of edge cases
            df['returns'] = df[price_col].pct_change()
            df['returns'] = df['returns'].replace([np.inf, -np.inf], np.nan)
            df['returns'] = df['returns'].ffill().bfill()
            
            # Calculate volatility with proper window
            df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std()
            df['volatility'] = df['volatility'].replace([np.inf, -np.inf], np.nan)
            df['volatility'] = df['volatility'].ffill().bfill()
            
            # Calculate RSI with proper handling of edge cases
            delta = df[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)  # Avoid division by zero
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi'] = df['rsi'].replace([np.inf, -np.inf], np.nan)
            df['rsi'] = df['rsi'].ffill().bfill()
            
            # Calculate MACD with proper handling of edge cases
            exp1 = df[price_col].ewm(span=12, adjust=False, min_periods=1).mean()
            exp2 = df[price_col].ewm(span=26, adjust=False, min_periods=1).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=1).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Calculate Bollinger Bands with proper handling of edge cases
            df['bb_middle'] = df[price_col].rolling(window=20, min_periods=1).mean()
            bb_std = df[price_col].rolling(window=20, min_periods=1).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Calculate ATR with proper handling of edge cases
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df[price_col].shift())
            low_close = np.abs(df['low'] - df[price_col].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr'] = true_range.rolling(14, min_periods=1).mean()
            
            # Calculate OBV with proper handling of edge cases
            df['obv'] = (np.sign(df[price_col].diff()) * df['volume']).fillna(0).cumsum()
            
            # Calculate ADX and DI with proper handling of edge cases
            # First calculate True Range
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df[price_col].shift())
            low_close = np.abs(df['low'] - df[price_col].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            
            # Calculate +DM and -DM with proper price movement detection
            high_diff = df['high'] - df['high'].shift(1)
            low_diff = df['low'].shift(1) - df['low']
            
            # Initialize +DM and -DM
            plus_dm = pd.Series(0.0, index=df.index)
            minus_dm = pd.Series(0.0, index=df.index)
            
            # Calculate +DM and -DM according to Wilder's method
            plus_dm = np.where(
                (high_diff > low_diff) & (high_diff > 0),
                high_diff,
                0
            )
            
            minus_dm = np.where(
                (low_diff > high_diff) & (low_diff > 0),
                low_diff,
                0
            )
            
            # Convert to Series
            plus_dm = pd.Series(plus_dm, index=df.index)
            minus_dm = pd.Series(minus_dm, index=df.index)
            
            # Calculate smoothed TR, +DM, and -DM using Wilder's smoothing method
            # First period uses simple average
            tr14 = pd.Series(index=df.index)
            plus_dm14 = pd.Series(index=df.index)
            minus_dm14 = pd.Series(index=df.index)
            
            # First period uses simple average
            tr14.iloc[13] = true_range.iloc[:14].mean()
            plus_dm14.iloc[13] = plus_dm.iloc[:14].mean()
            minus_dm14.iloc[13] = minus_dm.iloc[:14].mean()
            
            # Subsequent periods use Wilder's smoothing method
            for i in range(14, len(df)):
                tr14.iloc[i] = tr14.iloc[i-1] - (tr14.iloc[i-1] / 14) + true_range.iloc[i]
                plus_dm14.iloc[i] = plus_dm14.iloc[i-1] - (plus_dm14.iloc[i-1] / 14) + plus_dm.iloc[i]
                minus_dm14.iloc[i] = minus_dm14.iloc[i-1] - (minus_dm14.iloc[i-1] / 14) + minus_dm.iloc[i]
            
            # Forward fill any NaN values at the start
            tr14 = tr14.ffill()
            plus_dm14 = plus_dm14.ffill()
            minus_dm14 = minus_dm14.ffill()
            
            # Calculate DI+ and DI- with proper handling of edge cases
            plus_di14 = 100 * (plus_dm14 / tr14.replace(0, np.nan))
            minus_di14 = 100 * (minus_dm14 / tr14.replace(0, np.nan))
            
            # Forward fill any NaN values in DI+ and DI-
            plus_di14 = plus_di14.ffill()
            minus_di14 = minus_di14.ffill()
            
            # Calculate DI+ and DI-
            df['di_plus'] = plus_di14
            df['di_minus'] = minus_di14
            
            # Calculate DX with proper handling of edge cases
            # Use the standard formula: DX = 100 * |DI+ - DI-| / (DI+ + DI-)
            dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14).replace(0, np.nan)
            
            # Calculate ADX using Wilder's smoothing method
            # First period uses simple average
            adx = pd.Series(index=df.index)
            adx.iloc[13] = dx.iloc[:14].mean()
            
            # Subsequent periods use Wilder's smoothing method
            for i in range(14, len(df)):
                adx.iloc[i] = adx.iloc[i-1] - (adx.iloc[i-1] / 14) + dx.iloc[i]
            
            # Forward fill any NaN values at the start
            adx = adx.ffill()
            
            # Store ADX in the dataframe
            df['adx'] = adx
            
            # Forward fill any remaining NaN values
            df = df.ffill()
            
            # Backward fill any remaining NaN values at the start
            df = df.bfill()
            
            # Verify no NaN values remain
            if df.isna().any().any():
                nan_cols = df.columns[df.isna().any()].tolist()
                logger.warning(f"NaN values remain in columns: {nan_cols}")
                # Fill any remaining NaNs with 0
                df = df.fillna(0)
            
            # Log statistics of calculated indicators
            logger.info("Technical indicator statistics:")
            for col in ['returns', 'volatility', 'rsi', 'macd', 'atr', 'adx']:
                if col in df.columns:
                    stats = df[col].describe()
                    logger.info(f"  {col}:")
                    logger.info(f"    - Mean: {stats['mean']:.4f}")
                    logger.info(f"    - Std: {stats['std']:.4f}")
                    logger.info(f"    - Min: {stats['min']:.4f}")
                    logger.info(f"    - Max: {stats['max']:.4f}")
            
            # Log sample of ADX calculation components
            logger.info("ADX calculation components:")
            logger.info(f"  True Range - Mean: {tr14.mean():.4f}, Std: {tr14.std():.4f}")
            logger.info(f"  +DM - Mean: {plus_dm14.mean():.4f}, Std: {plus_dm14.std():.4f}")
            logger.info(f"  -DM - Mean: {minus_dm14.mean():.4f}, Std: {minus_dm14.std():.4f}")
            logger.info(f"  DI+ - Mean: {plus_di14.mean():.4f}, Std: {plus_di14.std():.4f}")
            logger.info(f"  DI- - Mean: {minus_di14.mean():.4f}, Std: {minus_di14.std():.4f}")
            logger.info(f"  DX - Mean: {dx.mean():.4f}, Std: {dx.std():.4f}")
            logger.info(f"  ADX - Mean: {df['adx'].mean():.4f}, Std: {df['adx'].std():.4f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return df

    def load_ticker_data(self, ticker: str) -> pd.DataFrame:
        """Load and enhance data for a single ticker with feature masking"""
        # Normalize ticker and path to lowercase
        ticker = ticker.strip().upper()
        agg_path = f'historical/{ticker}/aggregates/minute/'
        logger = logging.getLogger("training")
        logger.debug("🔄 Loading data from normalized path: %s", agg_path)
        
        logger.debug("\U0001F3C1 Starting data loading pipeline for %s", ticker)
        
        # Load raw OHLCV data with validation
        ohlcv = self._load_s3_data(agg_path)
        if ohlcv is None or ohlcv.empty:
            logger.warning("\U0001F6A8 No data loaded for %s - returning None", ticker)
            return None
            
        logger.debug("\U0001F4C5 Loaded raw data shape: %s", ohlcv.shape)
        
        # Corporate action merge with exception handling
        try:
            logger.debug("\U0001F4B2 Merging corporate actions...")
            ohlcv = self._merge_corporate_actions(ohlcv, ticker)
        except Exception as e:
            logger.error("\U0001F4A5 Corporate action merge failed: %s", str(e))
            return None
        
        # Add quote features
        quotes = self._process_quotes(ticker)
        
        merged = pd.merge(
            ohlcv,
            quotes,
            left_index=True,
            right_index=True,
            how='outer',
            sort=True
        ).ffill().dropna()  # Forward fill and remove remaining NaNs

        # Calculate technical indicators
        logger.debug("📊 Calculating technical indicators...")
        merged = self._calculate_technical_indicators(merged)

        # After merging in load_ticker_data()
        logger.info(f"📊 Post-merge stats for {ticker}:")
        logger.info(f" - Rows: {len(merged)}")
        logger.info(f" - NaNs: {merged.isna().sum().sum()}")
        logger.info(f" - Date range: {merged.index.min()} to {merged.index.max()}")
        
        # Log sample of technical indicators
        logger.info("Sample of technical indicators:")
        for col in ['returns', 'volatility', 'rsi', 'macd', 'atr', 'adx']:
            if col in merged.columns:
                sample = merged[col].head(5)
                logger.info(f"  {col}: {sample.values}")

        # Create default spread and mid_price columns if missing
        if 'bid_ask_spread' not in merged:
            merged['bid_ask_spread'] = 0.001  # Default spread
            logger.warning(f"Using default spreads for {ticker}")
        
        if 'mid_price' not in merged:
            merged['mid_price'] = merged['close']  # Use close price as mid price
            logger.warning(f"Using close price as mid price for {ticker}")

        # Apply feature mask if available
        if self.feature_mask is not None:
            available_features = [col for col in self.feature_columns if col in merged.columns]
            missing_features = set(self.feature_columns) - set(available_features)
            if missing_features:
                logger.warning(f"Missing features for {ticker}: {missing_features}")
                for feature in missing_features:
                    merged[feature] = 0.0  # Default value for missing features
            
            return merged[self.feature_columns]
        else:
            return merged[[col for col in self.all_feature_columns if col in merged.columns]]

    def _load_s3_data(self, prefix: str) -> pd.DataFrame:
        """Parallel S3 loading with validation and memory management"""
        logger = logging.getLogger("training")
        logger.debug(f"🔎 Loading from s3://{self.bucket}/{prefix}")
        
        s3 = boto3.client('s3',
                        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-west-2'),
                        config=Config(signature_version='s3v4', max_pool_connections=50))
        
        # Get all Parquet keys first
        keys = []
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('/') or not key.lower().endswith('.parquet'):
                    logger.debug(f"📦 Processing S3 object: {obj['Key']} (Size: {obj['Size']} bytes)")
                    continue
                keys.append(key)
        
        if not keys:
            logger.warning(f"No parquet files found in s3://{self.bucket}/{prefix}")
            return pd.DataFrame()
        
        # Parallel processing function with validation
        def process_key(key):
            try:
                logger.info(f"📥 Downloading {key}")
                response = s3.get_object(Bucket=self.bucket, Key=key)
                bio = BytesIO(response['Body'].read())

                # Split dtype mappings by data type
                ORDERBOOK_DTYPES = {
                    'bid_price': 'float32',
                    'ask_price': 'float32',
                    'bid_size': 'uint16',
                    'ask_size': 'uint16',
                    'sip_timestamp': 'int64'
                }

                OHLCV_DTYPES = {
                    'open': 'float32',
                    'high': 'float32',
                    'low': 'float32',
                    'close': 'float32',
                    'volume': 'uint16',
                    'vwap': 'float32'
                }

                # Determine data type and required columns
                is_quote_data = 'quotes' in key
                if is_quote_data:
                    required_columns = {'bid_price', 'ask_price', 'bid_size', 'ask_size', 'sip_timestamp'}
                    dtypes = ORDERBOOK_DTYPES
                else:
                    required_columns = {'open', 'high', 'low', 'close', 'volume', 'vwap'}
                    dtypes = OHLCV_DTYPES

                df = pd.read_parquet(bio)
                if required_columns and not required_columns.issubset(df.columns):
                    raise ValueError(f"Missing columns: {required_columns - set(df.columns)}")
                    
                # Apply dtype conversions
                df = df.astype({k:v for k,v in dtypes.items() if k in df.columns})
                
                # Convert index to datetime if it's not already
                if not isinstance(df.index, pd.DatetimeIndex):
                    try:
                        # Try to parse the filename for the date
                        date_str = key.split('/')[-1].replace('.parquet', '')
                        df.index = pd.to_datetime(df.index, unit='s') + pd.Timestamp(date_str)
                    except:
                        # If that fails, try to parse the index directly
                        df.index = pd.to_datetime(df.index)
                
                # Log sample of loaded data
                logger.info(f"Sample of loaded data from {key}:")
                logger.info(f"Shape: {df.shape}")
                logger.info(f"Columns: {df.columns.tolist()}")
                if is_quote_data:
                    logger.info(f"First 5 rows of bid/ask prices:")
                    logger.info(f"  bid_price: {df['bid_price'].head().values}")
                    logger.info(f"  ask_price: {df['ask_price'].head().values}")
                else:
                    logger.info(f"First 5 rows of close prices: {df['close'].head().values}")
                
                logger.info(f"✅ Loaded {len(df)} rows from {key}")
                return df
                
            except Exception as e:
                logger.error(f"🔥 Failed {key}: {str(e)}")
                return pd.DataFrame()

        # Parallel execution with memory management
        dfs = []
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = [executor.submit(process_key, key) for key in keys]
            for future in futures:
                df = future.result()
                if not df.empty:
                    dfs.append(df)
                    # Maintain memory cleanup
                    if len(dfs) % 10 == 0:
                        pd.concat(dfs).reset_index(drop=True)
                        gc.collect()
        
        if not dfs:
            logger.warning(f"No data loaded from s3://{self.bucket}/{prefix}")
            return pd.DataFrame()
            
        final_df = pd.concat(dfs, ignore_index=False).sort_index()
        logger.info(f"✅ Loaded {len(final_df)} total records")
        
        # Log sample of final data
        logger.info("Sample of final data:")
        logger.info(f"Shape: {final_df.shape}")
        logger.info(f"Columns: {final_df.columns.tolist()}")
        if 'close' in final_df.columns:
            logger.info(f"First 5 rows of close prices: {final_df['close'].head().values}")
        elif 'bid_price' in final_df.columns:
            logger.info(f"First 5 rows of bid/ask prices:")
            logger.info(f"  bid_price: {final_df['bid_price'].head().values}")
            logger.info(f"  ask_price: {final_df['ask_price'].head().values}")
        
        return final_df
    
    def create_tf_dataset(self, data: pd.DataFrame, window: int = 60) -> tf.data.Dataset:
        """Create validated sequences with strict shape enforcement and feature masking"""
        ds = tf.data.Dataset.from_generator(
            lambda: self._sequence_generator(data, window),
            output_signature=tf.TensorSpec(
                shape=(window, len(self.feature_columns)),
                dtype=tf.float32
            )
        )
        # Optimize dataset for GPU
        ds = ds.cache()  # Cache the dataset in memory
        ds = ds.batch(4096).prefetch(tf.data.AUTOTUNE)
        return ds
    
    def _sequence_generator(self, data, window):
        """Yield only valid sequences with feature masking"""
        if data is None or len(data) == 0:
            raise ValueError("Input data cannot be None or empty for sequence generation")
        
        num_features = len(self.feature_columns)
        
        for i in range(window, len(data)):
            seq = data.iloc[i-window:i][self.feature_columns].values
            
            if seq.shape != (window, num_features):
                logging.debug(f"Skipping invalid sequence at index {i} with shape {seq.shape}")
                continue
                
            yield seq

    def _merge_corporate_actions(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Merge corporate actions data with safety checks and fallbacks"""
        logger = logging.getLogger("training")
        if df.empty:
            logger.warning(f"Empty DataFrame received for {ticker} - skipping corporate actions")
            return df

        try:
            # Initialize default values first
            df = df.copy()
            df['days_since_dividend'] = 3650
            df['split_ratio'] = 1.0

            # Load corporate actions if available
            ca = self.corporate_actions.get(ticker.upper(), pd.DataFrame())
            if ca.empty:
                logger.debug(f"No corporate actions found for {ticker}")
                return df

            # Add missing columns with defaults instead of erroring
            required_ca_cols = {'ex_date', 'type', 'payment_date', 'ratio'}
            for col in required_ca_cols:
                if col not in ca.columns:
                    ca[col] = pd.NaT if 'date' in col else (1.0 if col == 'ratio' else np.nan)
                    logger.warning(f"Added missing corporate action column {col} with defaults for {ticker}")

            # Process dividends
            if 'dividend' in ca['type'].values:
                try:
                    dividends = ca[ca['type'] == 'dividend'].sort_values('ex_date').dropna(subset=['ex_date'])
                    if not dividends.empty:
                        last_div_date = dividends['ex_date'].max()
                        df['days_since_dividend'] = (df.index - last_div_date).dt.days.clip(upper=3650)
                        logger.debug(f"Set dividend dates for {ticker}")
                except Exception as e:
                    logger.error(f"Dividend processing failed: {str(e)}")

            # Process splits
            if 'split' in ca['type'].values:
                try:
                    splits = ca[ca['type'] == 'split'].sort_values('ex_date').dropna(subset=['ex_date'])
                    if not splits.empty:
                        last_split_date = splits['ex_date'].max()
                        df.loc[df.index >= last_split_date, 'split_ratio'] = splits['ratio'].iloc[-1]
                except Exception as e:
                    logger.error(f"Split processing failed: {str(e)}")

            return df.fillna({'days_since_dividend': 3650, 'split_ratio': 1.0})

        except Exception as e:
            logger.error(f"Corporate action merge failed: {str(e)}")
            return df  # Return DF with safe defaults

    def _process_quotes(self, ticker: str) -> pd.DataFrame:
        """Process raw quote data into spread features with robust error handling"""
        logger = logging.getLogger("training")
        logger.info(f"📈 Processing quotes for {ticker}")
        
        try:
            # Load raw quotes
            quotes = self._load_s3_data(f'historical/{ticker}/quotes/')
            
            if quotes.empty:
                logger.warning(f"⚠️ No quote data found for {ticker}")
                # Create empty DataFrame with required columns
                return pd.DataFrame(columns=['bid_price', 'ask_price', 'bid_size', 'ask_size', 
                                          'bid_ask_spread', 'mid_price'])

            # Validate quote schema
            required_columns = {
                'bid_price', 'ask_price', 
                'bid_size', 'ask_size', 
                'sip_timestamp'
            }
            missing_cols = required_columns - set(quotes.columns)
            
            if missing_cols:
                logger.error(f"🚫 Missing quote columns: {missing_cols}")
                # Create empty DataFrame with required columns
                return pd.DataFrame(columns=['bid_price', 'ask_price', 'bid_size', 'ask_size', 
                                          'bid_ask_spread', 'mid_price'])

            # Convert timestamp
            quotes['timestamp'] = pd.to_datetime(
                quotes['sip_timestamp'], unit='ns', errors='coerce'
            ).dropna()
            
            if quotes['timestamp'].isnull().any():
                logger.warning(f"⏰ Invalid timestamps in {len(quotes)} quotes")

            # Resample to 1-minute bars
            quotes.set_index('timestamp', inplace=True)
            
            # Filter out zero or invalid prices
            valid_quotes = quotes[
                (quotes['bid_price'] > 0) & 
                (quotes['ask_price'] > 0) & 
                (quotes['bid_price'] < quotes['ask_price'])
            ]
            
            if len(valid_quotes) == 0:
                logger.warning("No valid quotes found after filtering")
                return pd.DataFrame(columns=['bid_price', 'ask_price', 'bid_size', 'ask_size', 
                                          'bid_ask_spread', 'mid_price'])
            
            resampled = valid_quotes.resample('1min').agg({
                'bid_price': 'last',  # Use last price instead of mean
                'ask_price': 'last',
                'bid_size': 'sum',
                'ask_size': 'sum'
            })
            
            # Calculate spread features and mid price
            quotes_processed = resampled.assign(
                bid_ask_spread=lambda x: x['ask_price'] - x['bid_price'],
                mid_price=lambda x: (x['ask_price'] + x['bid_price']) / 2
            ).dropna()
            
            # Forward fill any missing values
            quotes_processed = quotes_processed.ffill()
            
            # Log sample of processed quotes
            logger.info("Sample of processed quotes:")
            logger.info(f"Shape: {quotes_processed.shape}")
            logger.info(f"Columns: {quotes_processed.columns.tolist()}")
            logger.info(f"First 5 rows of bid/ask prices:")
            logger.info(f"  bid_price: {quotes_processed['bid_price'].head().values}")
            logger.info(f"  ask_price: {quotes_processed['ask_price'].head().values}")
            logger.info(f"  spread: {quotes_processed['bid_ask_spread'].head().values}")
            
            logger.info(f"✅ Processed {len(quotes_processed)} quote bars")
            return quotes_processed

        except Exception as e:
            logger.error(f"🔥 Quote processing failed: {str(e)}")
            logger.debug("Stack trace:", exc_info=True)
            # Return empty DataFrame with required columns
            return pd.DataFrame(columns=['bid_price', 'ask_price', 'bid_size', 'ask_size', 
                                      'bid_ask_spread', 'mid_price'])

    def create_sequences(self, data: pd.DataFrame, sequence_length: int = 60) -> np.ndarray:
        """Create strictly uniform sequences with feature masking"""
        logger = logging.getLogger("training")
        logger.info(f"Creating sequences with length {sequence_length} from data shape {data.shape}")
        
        if data.empty:
            logger.warning("Empty DataFrame provided for sequence creation")
            return np.array([], dtype=np.float32)
            
        if len(data) < sequence_length:
            logger.warning(f"Insufficient data for sequences. Need {sequence_length}, got {len(data)}")
            return np.array([], dtype=np.float32)
            
        sequences = []
        for i in range(sequence_length, len(data)):
            # Use feature_columns which may be masked
            seq = data.iloc[i-sequence_length:i][self.feature_columns].values
            if seq.shape != (sequence_length, len(self.feature_columns)):
                logger.warning(f"Invalid sequence shape: {seq.shape}, expected {(sequence_length, len(self.feature_columns))}")
                continue  # Skip invalid sequences
            sequences.append(seq)
            
        result = np.array(sequences, dtype=np.float32)  # Explicit dtype
        logger.info(f"Created {len(result)} sequences with shape {result.shape}")
        return result

    def load_model_metadata(self, model_path: str) -> Dict:
        """Load feature metadata from saved model"""
        try:
            metadata_path = f"{model_path}.metadata.npz"
            if os.path.exists(metadata_path):
                metadata = np.load(metadata_path, allow_pickle=True)
                self.feature_mask = metadata['feature_mask'].tolist()
                self.feature_columns = metadata['selected_features'].tolist()
                logging.info(f"Loaded feature metadata: {self.feature_columns}")
                return dict(metadata)
            return None
        except Exception as e:
            logging.error(f"Error loading model metadata: {str(e)}")
            return None
