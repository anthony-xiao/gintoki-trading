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
        
        merged = pd.merge(ohlcv, quotes, left_index=True, right_index=True, how='left')

            # Create default spread columns if missing
        if 'bid_ask_spread' not in merged:
            merged['bid_ask_spread'] = 0.001  # Default spread
            merged['mid_price'] = merged['close']
            logger.warning(f"Using default spreads for {ticker}")
        # return pd.merge(ohlcv, quotes, left_index=True, right_index=True, how='left')
        return merged[[col for col in self.feature_columns if col in merged.columns]]


        # In data_loader.py
    # def _load_s3_data(self, prefix: str) -> pd.DataFrame:
        """Load and concatenate parquet files from S3"""
        s3_uri = f"s3://{self.bucket}/{prefix}"
        logger = logging.getLogger("training")
        logger.debug(f"📂 Entering _load_s3_data for prefix: {s3_uri}")

        dfs = []
        paginator = self.s3.get_paginator('list_objects_v2')
        logger.debug(f"🔄 Starting S3 pagination for {prefix}")
        logger.debug(f"Using S3 bucket: {self.bucket}")
        logger.debug(f"Loading data from path: {prefix}")
        page_iterator = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
        logger.debug("\U0001F680 Established S3 connection successfully, starting data retrieval")
        try:

            for page in page_iterator:
                logger.debug(f"📄 Processing page with {len(page.get('Contents', []))} objects")
                # Fix 1: Handle empty pages gracefully
                objects = page.get('Contents', [])
                logger.debug(f"test log {objects}")
                if not objects:  # Check if objects list is empty
                    logger.warning(f"⚠️ Empty page for prefix {s3_uri},{page}")
                    continue  # Skip empty pages but don't warn

                # Fix 2: Process even if 'Contents' key exists but list is empty
                for obj in objects:
                    if obj['Key'].endswith('.parquet'):
                        logger.debug(f"📦 Processing S3 object: {obj['Key']} (Size: {obj['Size']} bytes)")
                        response = self.s3.get_object(Bucket=self.bucket, Key=obj['Key'])
                        dfs.append(pd.read_parquet(BytesIO(response['Body'].read())))
                    logger.error(f"\U0001F6A8 CRITICAL - No parquet files found in {prefix}")
                    raise ValueError(f"No data files found at s3://{self.bucket}/{prefix}")
                
                combined = pd.concat(dfs).sort_index()
                logger.info("\U0001F4C8 Loaded %d files with %d total rows from %s", 
                            len(dfs), len(combined), prefix)
                if combined.empty:
                    logger.error("\U0001F4A5 Concatenated data is empty for %s", prefix)
                    raise ValueError("Empty data after concatenation")
                return combined
        except Exception as e:
            logger.error(f"S3 load error: {str(e)}")
            raise

    def _load_s3_data(self, prefix: str) -> pd.DataFrame:
        """Load and concatenate Parquet files from S3 with robust error handling"""
        logger = logging.getLogger("training")
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
                                required_columns = {'bid_price', 'ask_price', 'bid_size', 'ask_size', 'bid_ask_spread'}
                            else:
                                required_columns = set()

                            df = pd.read_parquet(bio)


                            # Validate schema for specific data types
                            if required_columns:
                                if not required_columns.issubset(df.columns):
                                    missing = required_columns - set(df.columns)
                                    raise ValueError(f"Missing columns in {key}: {missing}")
                            
                            # With EXPLICIT dtype mapping:
                            dtype_map = {
                                'bid_ask_spread': 'float16',
                                'mid_price': 'float32',
                                'volume': 'uint32',
                                'high': 'float32',
                                'low': 'float32',
                                'open': 'float32',
                                'close': 'float32',
                                'vwap': 'float32'
                            }
                            df = df.astype(dtype_map)

                            # Downsample if needed
                            if len(df) > 1_000_000:
                                df = df.sample(1_000_000)
                            
                            dfs.append(df)

                             # Clear memory every 10 files
                            if len(dfs) % 10 == 0:
                                pd.concat(dfs, ignore_index=True).reset_index(drop=True)
                                
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

    # def _process_quotes(self, ticker: str) -> pd.DataFrame:
        """Process raw quotes into spread features"""
        logger = logging.getLogger("training")
        logger.debug("\U0001F3C3 Starting quote processing for %s", ticker)
        try:
            quotes = self._load_s3_data(f'historical/{ticker}/quotes/')
        except Exception as e:
            logger.error("\U0001F4DB Quote loading failed for %s: %s", ticker, str(e))
            raise

        if quotes.empty:
            return pd.DataFrame(columns=['bid_price', 'ask_price', 'bid_size', 'ask_size'], 
                            index=pd.DatetimeIndex([]))
        
        required_quote_cols = {'bid_price', 'ask_price', 'bid_size', 'ask_size'}
        if not required_quote_cols.issubset(quotes.columns):
            missing = required_quote_cols - set(quotes.columns)
            logger.error(f"Missing quote columns: {missing}")
            return pd.DataFrame()

        return quotes.resample('1min').agg({
            'bid_price': 'mean',
            'ask_price': 'mean',
            'bid_size': 'sum', 
            'ask_size': 'sum'
        }).assign(
            bid_ask_spread=lambda x: x['ask_price'] - x['bid_price'],
            mid_price=lambda x: (x['ask_price'] + x['bid_price']) / 2
        ).dropna()

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
            
            logger.success(f"✅ Processed {len(quotes_processed)} quote bars")
            return quotes_processed

        except Exception as e:
            logger.error(f"🔥 Quote processing failed: {str(e)}")
            logger.debug("Stack trace:", exc_info=True)
            return pd.DataFrame()  # Return empty to continue pipeline

    def create_sequences(self, data: pd.DataFrame, window: int = 60) -> np.ndarray:
        """Convert DataFrame to LSTM input sequences"""
        return np.array([
            data.iloc[i-window:i][self.feature_columns].values
            for i in range(window, len(data))
        ])
