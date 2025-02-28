import os
from dotenv import load_dotenv
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from multiprocessing.pool import ThreadPool
import logging
from typing import Dict, List
import boto3
from io import BytesIO
from botocore.exceptions import ClientError
import gc



# Local imports moved to function scope to prevent circular dependencies
from src.py.util.api_client import fetch_paginated_data  # New shared module

# Configure logging
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('data_ingestion.log'), logging.StreamHandler()]
)

# Constants
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
BASE_URL = "https://api.polygon.io"
MAX_THREADS = 8
TRADING_DAYS = set()  # Populated during initialization
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-west-2')
)

def upload_parquet_to_s3(df: pd.DataFrame, bucket: str, key: str) -> bool:
    """Upload DataFrame to S3 as Parquet"""
    try:
        buffer = BytesIO()
        df.to_parquet(buffer, engine='pyarrow', compression='snappy')
        buffer.seek(0)
        
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=buffer,
            ContentType='application/parquet'
        )
        return True
    except ClientError as e:
        logging.error(f"S3 upload failed: {str(e)}")
        return False

def is_trading_day(date: datetime) -> bool:
    """Check if a date is a trading day."""
    return date.strftime('%Y-%m-%d') in TRADING_DAYS


def fetch_aggregates(ticker: str, start: datetime, end: datetime,
                    multiplier: int = 1, timespan: str = "minute") -> pd.DataFrame:
    """Fetch OHLCV + VWAP data with null checks"""
    logging.info("state fetch aggregates")
    all_data = []
    
    current_start = start
    while current_start < end:
        batch_end = min(current_start + timedelta(days=30), end)
        
        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{current_start.date().isoformat()}/{batch_end.date().isoformat()}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        
        try:
            batch_data = fetch_paginated_data(url, params)
            # Validate each item before adding
            valid_items = [
                item for item in batch_data 
                if all(key in item for key in ['t', 'o', 'h', 'l', 'c', 'v'])
            ]
            all_data.extend(valid_items)
            
        except Exception as e:
            logging.error(f"Batch {current_start} to {batch_end} failed: {str(e)}")
            
        current_start = batch_end + timedelta(days=1)

    if not all_data:
        return pd.DataFrame()
    
    return pd.DataFrame(all_data).rename(columns={
        't': 'timestamp',
        'o': 'open',
        'h': 'high',
        'l': 'low', 
        'c': 'close',
        'v': 'volume',
        'vw': 'vwap'
    }).assign(
        timestamp=lambda df: df['timestamp'].astype(str)  # Convert to string early
    )

# def fetch_aggregates(ticker: str, start: datetime, end: datetime, 
#                     multiplier: int = 1, timespan: str = "minute") -> pd.DataFrame:
#     """Fetch OHLCV + VWAP data"""
#     url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start.date().isoformat()}/{end.date().isoformat()}"
#     params = {"adjusted": "true", "sort": "asc", "limit": 50000}
#     logging.info(f"Fetching {timespan} aggregates from {start} to {end}")

#     data = fetch_paginated_data(url, params)

#     # new code for monthly batches

#     # """Fetch OHLCV + VWAP data with monthly batches"""
#     # all_data = []
    
#     # # Split into monthly batches
#     # current_start = start
#     # while current_start < end:
#     #     batch_end = min(
#     #         current_start + timedelta(days=30),  # ~1 month
#     #         end
#     #     )
        
#     #     url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/" \
#     #           f"{current_start.date().isoformat()}/{batch_end.date().isoformat()}"
#     #     params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        
#     #     logging.info(f"Fetching {current_start.date()} to {batch_end.date()} for {ticker}")
#     #     batch_data = fetch_paginated_data(url, params)

        
#     #     all_data.extend(batch_data)
        
#     #     current_start = batch_end + timedelta(days=1)

#     # # new code for monthly batches ends 

#     if not data:
#         return pd.DataFrame()
    
#     df = pd.DataFrame(data).rename(columns={
#         "t": "timestamp", "o": "open", "h": "high", "n":"transactions",
#         "l": "low", "c": "close", "v": "volume", "vw": "vwap"
#     })
#     df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
#     return df.set_index("timestamp")


def fetch_splits(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch splits with optimized parameters"""
    start_time = time.time()
    logging.info(f"🔄 Starting splits fetch for {ticker}")
    
    try:
        url = f"{BASE_URL}/v3/reference/splits"
        params = {
            "ticker": ticker,
            "execution_date.gte": start_date,
            "execution_date.lte": end_date,
            "limit": 1000
        }
        
        data = fetch_paginated_data(url, params)
        if not data:
            logging.info(f"✅ No splits found for {ticker}")
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        df["execution_date"] = pd.to_datetime(df["execution_date"])
        logging.info(f"⏱️ Fetched {len(data)} splits in {time.time()-start_time:.2f}s")
        return df[["execution_date", "split_from", "split_to"]]
        
    except Exception as e:
        logging.error(f"❌ Split fetch failed after {time.time()-start_time:.2f}s: {str(e)}")
        return pd.DataFrame()

def fetch_dividends(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch dividends with proper URL format"""
    start_time = time.time()
    logging.info(f"🔄 Starting dividends fetch for {ticker}")
    
    try:
        url = f"{BASE_URL}/v3/reference/dividends"
        params = {
            "ticker": ticker,
            "ex_dividend_date.gte": start_date,
            "ex_dividend_date.lte": end_date,
            "limit": 1000
        }
        
        data = fetch_paginated_data(url, params)
        if not data:
            return pd.DataFrame()
        
        logging.info(f"⏱️ Fetched {len(data)} dividends in {time.time()-start_time:.2f}s")
        df = pd.DataFrame(data)
        df["ex_dividend_date"] = pd.to_datetime(df["ex_dividend_date"])
        return df[["ex_dividend_date", "cash_amount", "declaration_date"]]
        
    except Exception as e:
        logging.error(f"❌ Dividends fetch failed after {time.time()-start_time:.2f}s: {str(e)}")
        return pd.DataFrame()
    
def fetch_trades(ticker: str, date: str) -> pd.DataFrame:
    """Fetch trades with proper URL format"""
    start_time = time.time()
    logging.info(f"🔄 Starting trades fetch for {ticker} on {date}")
    
    try:
        url = f"{BASE_URL}/v3/trades/{ticker}"
        params = {
            "timestamp.gte": f"{date}T00:00:00.000Z",
            "timestamp.lte": f"{date}T23:59:59.999Z",
            "limit": 50000,
            "sort": "timestamp",
            "order": "asc"
        }
        
        data = fetch_paginated_data(url, params)
        
        if not data:
            logging.info(f"✅ No trades found for {ticker} on {date}")
            return pd.DataFrame()
        
        logging.info(f"⏱️ Fetched {len(data)} trades in {time.time()-start_time:.2f}s")
        df = pd.DataFrame(data)
        
        # Validate required columns
        required_columns = ["sip_timestamp", "price", "size"]
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Missing required trade column: {col}")
        
        # Handle optional columns
        optional_columns = ["conditions"]
        keep_columns = required_columns.copy()
        for col in optional_columns:
            if col in df.columns:
                keep_columns.append(col)
        
        df["timestamp"] = pd.to_datetime(df["sip_timestamp"], utc=True)
        logging.info(f"⏱️ Fetched {len(df)} trades in {time.time()-start_time:.2f}s")
        return df[keep_columns]
        
    except Exception as e:
        logging.error(f"❌ Trades fetch failed after {time.time()-start_time:.2f}s: {str(e)}")
        return pd.DataFrame()


def fetch_quotes(ticker: str, date: str) -> pd.DataFrame:
    """Fetch quotes with proper URL format"""
    start_time = time.time()
    logging.info(f"🔄 Starting quotes fetch for {ticker} on {date}")
    
    try:
        url = f"{BASE_URL}/v3/quotes/{ticker}"
        params = {
            "timestamp.gte": f"{date}T00:00:00.000Z",
            "timestamp.lte": f"{date}T23:59:59.999Z",
            "limit": 50000,
            "sort": "timestamp",
            "order": "asc"
        }
        
        data = fetch_paginated_data(url, params)
        
        # if not data:
        #     logging.info(f"✅ No quotes found for {ticker} on {date}")
        #     return pd.DataFrame()

        if not isinstance(data, list) or len(data) == 0:
            logging.warning(f"No quote data for {ticker} on {date}")
            return pd.DataFrame()
        
        logging.info(f"⏱️ Fetched {len(data)} quotes in {time.time()-start_time:.2f}s")
        df = pd.DataFrame(data)
        
        # Handle missing columns
        expected_columns = {
            "sip_timestamp": True,
            "bid_price": True,
            "bid_size": True,
            "ask_price": True,
            "ask_size": True,
            "indicators": False  # Optional
        }
        
        for col, required in expected_columns.items():
            if col not in df.columns and required:
                raise KeyError(f"Missing required column: {col}")
        
        df["timestamp"] = pd.to_datetime(df["sip_timestamp"], utc=True)
        keep_columns = [col for col in expected_columns if col in df.columns]
        
        logging.info(f"⏱️ Fetched {len(df)} quotes in {time.time()-start_time:.2f}s")
        return df[keep_columns]
        
    except Exception as e:
        logging.error(f"❌ Quotes fetch failed after {time.time()-start_time:.2f}s: {str(e)}")
        return pd.DataFrame()

def initialize_trading_days():
    """Self-contained trading calendar initialization"""
    global TRADING_DAYS
    TRADING_DAYS.clear()

    try:
        url = f"{BASE_URL}/v1/marketstatus/upcoming"
        response = requests.get(url, params={"apiKey": POLYGON_API_KEY})
        response.raise_for_status()
        
        market_status = response.json()
        start_date = "2000-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        all_dates = pd.date_range(start_date, end_date, freq='B')
        holidays = {
            datetime.strptime(d["date"], "%Y-%m-%d").date()
            for d in market_status if d.get("status") == "closed"
        }
        
        TRADING_DAYS.update(
            date.date().isoformat() 
            for date in all_dates 
            if date.date() not in holidays
        )
        logging.info(f"Initialized {len(TRADING_DAYS)} trading days")
        
    except Exception as e:
        logging.error(f"Calendar init failed: {str(e)}")
        TRADING_DAYS.update(
            d.date().isoformat() 
            for d in pd.date_range("2000-01-01", datetime.now(), freq='B')
        )
        logging.warning("Using fallback weekday calendar")

def fetch_all_data(ticker: str, start_date: str, end_date: str) -> Dict[str, str]:
    """Main fetch function with deferred corporate actions import"""
    # Deferred import breaks circular dependency
    from src.py.util.corporate_actions import corporate_actions_manager
    """Fetch all data for a ticker."""
    results = {}
    logging.info("fetch all data start")
    try:
        # Fetch corporate actions first
        logging.info("\n=== Processing %s ===", ticker)
        
        # Get corporate actions
        logging.info("Fetching corporate actions...")
        bucket = corporate_actions_manager.fetch_corporate_actions(ticker, start_date, end_date)
        
        # Upload corporate actions
        s3_path = corporate_actions_manager.upload_corporate_actions_to_s3(
            bucket, ticker, start_date, end_date
        )
        
        if not s3_path:
            logging.warning("No corporate actions found for %s", ticker)
            s3_path = f"s3://{bucket}/historical/{ticker}/corporate_actions/empty.parquet"

        logging.info("Corporate actions path: %s", s3_path)



        # logging.info("fetch corporate action start")
        # # Initialize corporate actions
        # ca_manager = corporate_actions_manager
        # ca_manager.fetch_corporate_actions([ticker], start_date, end_date)

        # # Add corporate actions upload
        # ca_manager.upload_corporate_actions_to_s3(
        #     os.getenv('AWS_S3_BUCKET'), 
        #     ticker,
        #     start_date,
        #     end_date
    # )
    
        # Aggregates collection
        for res in [("minute", 1), ("day", 1)]:
            df = fetch_aggregates(
                ticker,
                datetime.strptime(start_date, "%Y-%m-%d"),
                datetime.strptime(end_date, "%Y-%m-%d"),
                res[1], res[0]
            )
            
            if not df.empty:
                # Local save (optional)
                # local_path = f"data/historical/{ticker}/aggregates_{res[0]}.parquet"
                # df.to_parquet(local_path, engine='pyarrow', compression='snappy')
                
                # S3 Upload
                logging.info(f"prepare to start upload aggregate {os.getenv('AWS_S3_BUCKET')}, {s3_path}")
                s3_path = f"historical/{ticker or 'unknown'}/aggregates/{res[0]}/{start_date}_to_{end_date}.parquet"
                logging.info(f"start upload aggregate {os.getenv('AWS_S3_BUCKET')}, {s3_path}")
                # s3_path = f"historical/{ticker}/aggregates/{res[0]}/{start_date}_to_{end_date}.parquet"
                if upload_parquet_to_s3(df, os.getenv('AWS_S3_BUCKET'), s3_path):
                    results[f"aggregates_{res[0]}"] = s3_path

        # Tick data collection
        dates = pd.date_range(start_date, end_date, freq="D")
        
        def process_date(date: datetime):
            date_str = date.strftime("%Y-%m-%d")
            if not is_trading_day(date):
                return (date_str, pd.DataFrame(), pd.DataFrame())
            return (
                date_str,
                fetch_trades(ticker, date_str),
                fetch_quotes(ticker, date_str)
            )

        with ThreadPool(MAX_THREADS) as pool:
            for date_str, trades, quotes in pool.imap(process_date, dates):

                if not trades.empty:
                    # Local save (optional)
                    # local_trade_path = f"data/historical/{ticker}/trades_{date_str}.parquet"
                    # trades.to_parquet(local_trade_path)
                    
                    # S3 Upload
                    s3_trade_path = f"historical/{ticker}/trades/{date_str}.parquet"
                    if upload_parquet_to_s3(trades, os.getenv('AWS_S3_BUCKET'), s3_trade_path):
                        results.setdefault("trades", []).append(s3_trade_path)
                        logging.info(f"Uploaded trades for {ticker} on {date_str} to s3://{os.getenv('AWS_S3_BUCKET')}/{s3_trade_path}")
                        # del trades  # Explicit deletion
                        # gc.collect()  # Force garbage collection

                if not quotes.empty:
                    # Local save (optional)
                    # local_quote_path = f"data/historical/{ticker}/quotes_{date_str}.parquet"
                    # quotes.to_parquet(local_quote_path)
                    
                    # S3 Upload  
                    s3_quote_path = f"historical/{ticker}/quotes/{date_str}.parquet"
                    if upload_parquet_to_s3(quotes, os.getenv('AWS_S3_BUCKET'), s3_quote_path):
                        results.setdefault("quotes", []).append(s3_quote_path)
                        logging.info(f"Uploaded quotes for {ticker} on {date_str} to s3://{os.getenv('AWS_S3_BUCKET')}/{s3_quote_path}")
                        # del quotes  # Explicit deletion
                        # gc.collect()  # Force garbage collection

    except Exception as e:
        logging.error(f"Critical error processing {ticker}: {str(e)}")
        
    return results

if __name__ == "__main__":
    import argparse
    from src.py.util.corporate_actions import corporate_actions_manager  # Absolute import

    parser = argparse.ArgumentParser(
        prog='historical_data_fetcher.py',
        description="Fetch Polygon.io historical data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--init-only", 
        action="store_true",
        help="Initialize trading calendar only"
    )
    parser.add_argument(
        "tickers",
        nargs="+",
        help="Space-separated list of stock tickers (e.g. AAPL MSFT)"
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=MAX_THREADS,
        help="Number of parallel processing threads"
    )
    
    args = parser.parse_args()

    # Initialize trading calendar first
    if not TRADING_DAYS:
        logging.info("Initializing trading calendar...")
        initialize_trading_days()

    if args.init_only:
        logging.info(f"Initialized {len(TRADING_DAYS)} trading days")
        exit(0)

    if not args.tickers:
        parser.error("At least one ticker symbol is required")

    # Validate date format
    try:
        datetime.strptime(args.start, "%Y-%m-%d")
        datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError:
        parser.error("Invalid date format. Use YYYY-MM-DD")

    # Initialize corporate actions manager
    ca_manager = corporate_actions_manager
    
    def process_ticker(ticker: str):
        """Wrapper function for error handling"""
        try:
            logging.info(f"v6 working 🚀 Starting data collection for {ticker}")
            start_time = time.time()
            
            result = fetch_all_data(ticker, args.start, args.end)
            elapsed = time.time() - start_time
            
            if result:
                logging.info(f"✅ Completed {ticker} in {elapsed:.2f}s "
                            f"({len(result)} datasets)")
                return True
            return False
            
        except Exception as e:
            logging.error(f"❌ Failed processing {ticker}: {str(e)}")
            return False

    # Process tickers in parallel
    with ThreadPool(min(args.threads, len(args.tickers))) as pool:
        results = pool.map(process_ticker, args.tickers)
        success_count = sum(results)

    # Fetch corporate actions after data collection
    logging.info("🔄 Collecting corporate actions for all symbols")
    ca_start = time.time()
    ca_manager.fetch_corporate_actions(args.tickers, args.start, args.end)
    logging.info(f"⏱️ Corporate actions collected in {time.time()-ca_start:.2f}s")

    logging.info(f"📊 Final result: {success_count}/{len(args.tickers)} "
                "tickers processed successfully")
