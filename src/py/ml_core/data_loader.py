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
    def __init__(self, bucket: str = 'quant-trader-data-gintoki'):
        logger = logging.getLogger("training")
        logger.debug("\U0001F310 Initializing S3 client for bucket %s", bucket)
        # self.s3 = boto3.client('s3')
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-west-2')
        )
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

        # After merging in load_ticker_data()
        logger.info(f"📊 Post-merge stats for {ticker}:")
        logger.info(f" - Rows: {len(merged)}")
        logger.info(f" - NaNs: {merged.isna().sum().sum()}")
        logger.info(f" - Date range: {merged.index.min()} to {merged.index.max()}")

            # Create default spread columns if missing
        if 'bid_ask_spread' not in merged:
            merged['bid_ask_spread'] = 0.001  # Default spread
            merged['mid_price'] = merged['close']
            logger.warning(f"Using default spreads for {ticker}")
        # return pd.merge(ohlcv, quotes, left_index=True, right_index=True, how='left')
        return merged[[col for col in self.feature_columns if col in merged.columns]]

    # def _load_s3_data(self, prefix: str) -> pd.DataFrame:
        """Load and concatenate Parquet files from S3 with robust error handling"""
        logger = logging.getLogger("training")
        logger.debug(f"🔎 Loading from s3://{self.bucket}/{prefix}")
        dfs = []
        
        try:
            logger.debug(f"🔎 Attempting to load data from S3 prefix: s3://{self.bucket}/{prefix}")
            
            # Initialize S3 client with explicit configuration
            s3 = boto3.client('s3',
                            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-west-2'),
                            config=Config(signature_version='s3v4'))
            
            paginator = s3.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket,
                Prefix=prefix,
                PaginationConfig={'PageSize': 1000}
            )

            for page_idx, page in enumerate(page_iterator, 1):
                logger.debug(f"📄 Processing page {page_idx}, KeyCount: {page.get('KeyCount', 0)}")
                
                if 'Contents' not in page:
                    logger.debug(f"ℹ️ No contents in page {page_idx}")
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    
                    # Skip directory markers and non-Parquet files
                    if key.endswith('/'):
                        logger.debug(f"🚫 Skipping directory marker: {key}")
                        continue
                        
                    if not key.lower().endswith('.parquet'):
                        logger.debug(f"🚫 Skipping non-Parquet file: {key}")
                        continue
                    
                    try:
                        logger.info(f"📥 Downloading {key} ({obj['Size']/1024/1024:.2f} MB)")
                        response = s3.get_object(Bucket=self.bucket, Key=key)
                        
                        # Validate Parquet file
                        with BytesIO(response['Body'].read()) as bio:
                            # df = pd.read_parquet(bio)

                            if 'aggregates' in key:
                                required_columns = {'open', 'high', 'low', 'close', 'volume', 'vwap'}
                            elif 'quotes' in key:
                                required_columns = {'bid_price', 'ask_price', 'bid_size', 'ask_size'}
                            else:
                                required_columns = set()

                            df = pd.read_parquet(bio)


                            # Validate schema for specific data types
                            if required_columns:
                                if not required_columns.issubset(df.columns):
                                    missing = required_columns - set(df.columns)
                                    raise ValueError(f"Missing columns in {key}: {sorted(missing)}")

                            # Split dtype mappings by data type
                            ORDERBOOK_DTYPES = {
                                'bid_price': 'float32',
                                'ask_price': 'float32',
                                'bid_size': 'uint16',
                                'ask_size': 'uint16'
                            }

                            OHLCV_DTYPES = {
                                'open': 'float32',
                                'high': 'float32',
                                'low': 'float32',
                                'close': 'float32',
                                'volume': 'uint16',
                                'vwap': 'float32'
                            }

                            # After reading the DF (line 229):
                            if 'bid_ask_spread' in df.columns:  # Order book data
                                df = df.astype(ORDERBOOK_DTYPES)
                            else:  # OHLCV data
                                df = df.astype({k:v for k,v in OHLCV_DTYPES.items() if k in df.columns})
                            

                            # Downsample if needed
                            if len(df) > 1_000_000:
                                df = df.sample(1_000_000)
                            
                            dfs.append(df)
                            
                            del df  # Explicit memory release
                            gc.collect()
                            df = pd.concat(dfs)

                             # Clear memory every 10 files
                            # if len(dfs) % 10 == 0:
                            #     pd.concat(dfs, ignore_index=True).reset_index(drop=True)
                                
                            logger.info(f"✅ Successfully loaded {len(df)} rows from {key}")

                    except Exception as e:
                        logger.error(f"🔥 Failed to process {key}: {str(e)}")
                        raise

            if not dfs:
                logger.critical(f"💥 No valid Parquet files found at s3://{self.bucket}/{prefix}")
                logger.debug("Verification command:")
                logger.debug(f"aws s3api list-objects-v2 --bucket {self.bucket} --prefix '{prefix}'")
                raise ValueError(f"No data loaded from s3://{self.bucket}/{prefix}")

            full_df = pd.concat(dfs).sort_index()
            logger.info(f"\U0001F4CA Loaded {len(full_df)} total rows from {len(dfs)} files")
            return full_df

        except ClientError as ce:
            logger.critical(f"🔐 AWS Client Error: {ce.response['Error']['Message']}")
            if ce.response['Error']['Code'] == 'AccessDenied':
                logger.error("❗ Verify IAM permissions and bucket policy")
            raise
        except Exception as e:
            logger.critical(f"💣 Critical error in S3 loading: {str(e)}", exc_info=True)
            raise


    # def _merge_corporate_actions(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Enrich data with corporate action features"""
        if df is None or df.empty:
            logger = logging.getLogger("training")
            logger.error("\U0001F525 Cannot merge corporate actions - input DF is empty")
            return df
            
        ca = self.corporate_actions.get(ticker, pd.DataFrame())
        logger = logging.getLogger("training")
        logger.debug("\U0001F4E6 Found %d corporate actions for %s", len(ca), ticker)
        
        try:
            # Dividend features
            if not ca.empty and 'dividend' in ca['type'].values:
                dividends = ca[ca['type'] == 'dividend']
                logger.debug("\U0001F4B0 Adding dividend features from %d events", len(dividends))
                df['days_since_dividend'] = df.index.map(
                    lambda x: (x - dividends[dividends['ex_date'] < x]['ex_date'].max()).days
                    if not dividends.empty else 3650
                ).fillna(3650)
            else:
                logger.debug("\U0001F4C4 No dividend actions found - setting default")
                df['days_since_dividend'] = 3650
        except Exception as e:
            logger.error("\U0001F525 Failed to merge corporate actions: %s", str(e))
            raise

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
                    'ask_size': 'uint16'
                }

                OHLCV_DTYPES = {
                    'open': 'float32',
                    'high': 'float32',
                    'low': 'float32',
                    'close': 'float32',
                    'volume': 'uint16',
                    'vwap': 'float32'
                }

                # Maintain original validation logic
                if 'aggregates' in key:
                    required_columns = {'open', 'high', 'low', 'close', 'volume', 'vwap'}
                    dtypes = OHLCV_DTYPES
                elif 'quotes' in key:
                    required_columns = {'bid_price', 'ask_price', 'bid_size', 'ask_size'}
                    dtypes = ORDERBOOK_DTYPES
                else:
                    required_columns = set()
                    dtypes = {}

                df = pd.read_parquet(bio)
                if required_columns and not required_columns.issubset(df.columns):
                    raise ValueError(f"Missing columns: {required_columns - set(df.columns)}")
                    
                # Apply dtype conversions
                df = df.astype({k:v for k,v in dtypes.items() if k in df.columns})
                
                # Downsample large files
                if len(df) > 1_000_000:
                    df = df.sample(1_000_000)
                    
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
        
        final_df = pd.concat(dfs, ignore_index=False).sort_index()
        logger.info(f"✅ Loaded {len(final_df)} total records")
        return final_df 

    # def create_tf_dataset(self, data: pd.DataFrame, window: int = 60) -> tf.data.Dataset:
        """Create uniform-length sequences with strict windowing"""
        ds = tf.data.Dataset.from_tensor_slices(data[self.feature_columns].values)
        # Enforce exact window size and convert to numpy-friendly format
        ds = ds.window(window, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window))  # Convert to tensors
        ds = ds.map(lambda x: tf.ensure_shape(x, [window, len(self.feature_columns)]))
        
        return ds.batch(4096).prefetch(tf.data.AUTOTUNE)
    
    def create_tf_dataset(self, data: pd.DataFrame, window: int = 60) -> tf.data.Dataset:
        """Create validated sequences with strict shape enforcement"""
        ds = tf.data.Dataset.from_generator(
            lambda: self._sequence_generator(data, window),
            output_signature=tf.TensorSpec(
                shape=(window, len(self.feature_columns)),
                dtype=tf.float32
            )
        )
        return ds.batch(4096).prefetch(tf.data.AUTOTUNE)
    
    def _sequence_generator(self, data, window):
        """Yield only valid sequences"""
        for i in range(window, len(data)):
            seq = data.iloc[i-window:i][self.feature_columns].values
            if seq.shape == (window, len(self.feature_columns)):
                yield seq


    # def _merge_corporate_actions(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
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

            # Validate corporate actions schema
            required_ca_cols = {'ex_date', 'type', 'payment_date'}
            if not required_ca_cols.issubset(ca.columns):
                missing = required_ca_cols - set(ca.columns)
                logger.error(f"Missing corporate action columns {missing} for {ticker}")
                return df

            # Process dividends
            if 'dividend' in ca['type'].values:
                try:
                    dividends = ca[ca['type'] == 'dividend'].sort_values('ex_date')
                    last_div_date = dividends['ex_date'].max()
                    df['days_since_dividend'] = (df.index - last_div_date).days.clip(upper=3650)
                    logger.debug(f"Set dividend dates for {ticker}")
                except Exception as e:
                    logger.error(f"Dividend processing failed: {str(e)}")

            # Process splits
            if 'split' in ca['type'].values:
                try:
                    splits = ca[ca['type'] == 'split'].sort_values('ex_date')
                    if not splits.empty:
                        last_split = splits.iloc[-1]
                        df['split_ratio'] = last_split['ratio'] if 'ratio' in splits.columns else 1.0
                        logger.debug(f"Applied split ratio {df['split_ratio'].iloc[0]} for {ticker}")
                except Exception as e:
                    logger.error(f"Split processing failed: {str(e)}")

            return df

        except Exception as e:
            logger.error(f"Corporate action merge failed: {str(e)}")
            return df  # Return original DF with default values

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
                return pd.DataFrame()

            # Validate quote schema
            required_columns = {
                'bid_price', 'ask_price', 
                'bid_size', 'ask_size', 
                'sip_timestamp'
            }
            missing_cols = required_columns - set(quotes.columns)
            
            if missing_cols:
                logger.error(f"🚫 Missing quote columns: {missing_cols}")
                return pd.DataFrame()

            # Convert timestamp
            quotes['timestamp'] = pd.to_datetime(
                quotes['sip_timestamp'], unit='ns', errors='coerce'
            ).dropna()
            
            if quotes['timestamp'].isnull().any():
                logger.warning(f"⏰ Invalid timestamps in {len(quotes)} quotes")

            # Resample to 1-minute bars
            quotes.set_index('timestamp', inplace=True)
            
            resampled = quotes.resample('1min').agg({
                'bid_price': 'mean',
                'ask_price': 'mean',
                'bid_size': 'sum',
                'ask_size': 'sum'
            })
            
            # Calculate spread features
            quotes_processed = resampled.assign(
                bid_ask_spread=lambda x: x['ask_price'] - x['bid_price'],
                mid_price=lambda x: (x['ask_price'] + x['bid_price']) / 2
            ).dropna()
            
            logger.info(f"✅ Processed {len(quotes_processed)} quote bars")
            return quotes_processed

        except Exception as e:
            logger.error(f"🔥 Quote processing failed: {str(e)}")
            logger.debug("Stack trace:", exc_info=True)
            return pd.DataFrame()  # Return empty to continue pipeline

    def create_sequences(self, data: pd.DataFrame, window: int = 60) -> np.ndarray:
        """Create strictly uniform sequences"""
        sequences = []
        for i in range(window, len(data)):
            seq = data.iloc[i-window:i][self.feature_columns].values
            if seq.shape != (window, len(self.feature_columns)):  # New validation
                continue  # Skip invalid sequences
            sequences.append(seq)
        return np.array(sequences, dtype=np.float32)  # Explicit dtype
