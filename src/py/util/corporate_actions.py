import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List
import logging
from src.py.data_ingestion.historical_data_fetcher import upload_parquet_to_s3

# Shared utilities
from src.py.util.api_client import fetch_paginated_data

# Constants
load_dotenv()
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
BASE_URL = "https://api.polygon.io"

class corporate_actions_manager:
    def __init__(self):
        self.splits = pd.DataFrame()
        self.dividends = pd.DataFrame()
        self.split_map = {}
        self.dividend_map = {}

    def fetch_corporate_actions(self, symbols: List[str], start_date: str, end_date: str):
        logging.info(f"start corporate fecthing {symbols}")
        """Fetch splits and dividends"""
        self._fetch_splits(symbols, start_date, end_date)
        self._fetch_dividends(symbols, start_date, end_date)
        self._create_adjustment_maps()

    def _fetch_splits(self, symbols: List[str], start_date: str, end_date: str):
        """Fetch stock splits"""
        url = f"{BASE_URL}/v3/reference/splits"
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        all_splits = []

        # New code 
        for year in range(start_year, end_year+1):
            year_start = max(start_date, f"{year}-01-01")
            year_end = min(end_date, f"{year}-12-31")
        
        # Old code
            params = {
                'ticker.in': ','.join(symbols),
                'execution_date.gte': start_date,
                'execution_date.lte': end_date,
                'limit': 1000
            }
        # new code
            logging.info(f"Fetching splits {year_start} to {year_end}")
            data = fetch_paginated_data(url, params)
            all_splits.extend(data)
        
        # data = fetch_paginated_data(url, params)

        if all_splits:
            if 'ticker' not in pd.DataFrame(all_splits).columns:
                logging.warning("No 'ticker' column in splits data ,{data}")
                self.splits = pd.DataFrame()
        else:
            self.splits = pd.DataFrame(all_splits)[['ticker', 'execution_date', 'split_from', 'split_to']]
            self.splits.rename(columns={'ticker': 'symbol'}, inplace=True)
            logging.info(f"splits {all_splits} ")
            logging.info(f"splits {self.splits} ")


        #     self.splits = pd.DataFrame(all_splits)[['ticker', 'execution_date', 'split_from', 'split_to']]
        #     self.splits.rename(columns={'ticker': 'symbol'}, inplace=True)
        # else:
        #     self.splits = pd.DataFrame()
        #     logging.info("No splits data found")


    def _fetch_dividends(self, symbols: List[str], start_date: str, end_date: str):
        """Fetch dividends"""
        url = f"{BASE_URL}/v3/reference/dividends"

        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        all_dividends = []

        for year in range(start_year, end_year+1):
            year_start = max(start_date, f"{year}-01-01")
            year_end = min(end_date, f"{year}-12-31")
        
    
            params = {
                'ticker.in': ','.join(symbols),
                'ex_dividend_date.gte': start_date,
                'ex_dividend_date.lte': end_date,
                'limit': 1000
            }

            logging.info(f"Fetching dividends {year_start} to {year_end}")
            data = fetch_paginated_data(url, params)
            all_dividends.extend(data)
        
        if all_dividends:
            if 'ticker' not in pd.DataFrame(all_dividends).columns:
                logging.info("No ticker column in dividends data found, {data}")
                self.dividends = pd.DataFrame()
        else:
            self.dividends = pd.DataFrame(data)[['ticker', 'ex_dividend_date', 'cash_amount']]
            self.dividends.rename(columns={'ticker': 'symbol'}, inplace=True)
            logging.info(f"splits {all_dividends} ")
            logging.info(f"splits {self.dividends} ")
        
        # data = fetch_paginated_data(url, params)
        # if data:
        #     self.dividends = pd.DataFrame(data)[['ticker', 'ex_dividend_date', 'cash_amount']]
        #     self.dividends.rename(columns={'ticker': 'symbol'}, inplace=True)

    def _create_adjustment_maps(self):
        """Create adjustment lookup structures"""
        if not self.splits.empty:
            self.splits['execution_date'] = pd.to_datetime(self.splits['execution_date'])
            self.splits['split_ratio'] = self.splits['split_to'] / self.splits['split_from']
            self.split_map = self.splits.groupby('symbol')[['execution_date', 'split_ratio']] \
                .apply(lambda x: x.set_index('execution_date')['split_ratio'].to_dict()) \
                .to_dict()

        if not self.dividends.empty:
            self.dividends['ex_dividend_date'] = pd.to_datetime(self.dividends['ex_dividend_date'])
            self.dividend_map = self.dividends.groupby('symbol')[['ex_dividend_date', 'cash_amount']] \
                .apply(lambda x: x.set_index('ex_dividend_date')['cash_amount'].to_dict()) \
                .to_dict()

    def apply_adjustments(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Apply corporate actions to data"""

        symbol = str(symbol) if symbol else 'unknown'
        #new code
        if data.empty or not isinstance(data, pd.DataFrame):
            return data
        
        #old
        logger = logging.getLogger(__name__)
        adjusted = data.copy()
        
        # Apply splits
        if symbol in self.split_map:
            logger.info(f"Found {len(self.split_map[symbol])} splits for {symbol}")
            for dt, ratio in self.split_map[symbol].items():
                mask = adjusted.index >= dt
                adjusted.loc[mask, ['open', 'high', 'low', 'close']] /= ratio
                adjusted.loc[mask, 'volume'] *= ratio
                
        # Apply dividends
        if symbol in self.dividend_map:
            logger.info(f"Found {len(self.dividend_map[symbol])} dividends for {symbol}")
            for dt, amount in self.dividend_map[symbol].items():
                mask = adjusted.index >= dt
                adjusted.loc[mask, ['open', 'high', 'low', 'close']] -= amount
                
        return adjusted
    
    # def upload_corporate_actions_to_s3(self, bucket: str, ticker: str, start_date: str, end_date: str):
    #     """Upload splits and dividends to S3"""
    #     from src.py.data_ingestion.historical_data_fetcher import upload_parquet_to_s3  # Late import

    #     """Add null check for ticker"""
    #     if not ticker or pd.isnull(ticker):
    #         logging.error("Invalid ticker for corporate actions upload")
    #         return
        
    #     # Upload splits
    #     if not self.splits.empty:
    #         splits_key = f"historical/{ticker}/corporate_actions/splits/{start_date}_to_{end_date}.parquet"
    #         if upload_parquet_to_s3(self.splits, bucket, splits_key):
    #             logging.info(f"Uploaded splits to s3://{bucket}/{splits_key}")
        
    #     # Upload dividends
    #     if not self.dividends.empty:
    #         dividends_key = f"historical/{ticker}/corporate_actions/dividends/{start_date}_to_{end_date}.parquet"
    #         if upload_parquet_to_s3(self.dividends, bucket, dividends_key):
    #             logging.info(f"Uploaded dividends to s3://{bucket}/{dividends_key}")

    def upload_corporate_actions_to_s3(self, bucket: str, ticker: str, start: str, end: str) -> str:
        """Upload corporate actions to S3 with proper error handling"""
        key = f"historical/{ticker}/corporate_actions/{start}_to_{end}.parquet"
        logger = logging.getLogger(__name__)
        logger.info(f"🔄 Attempting to upload corporate actions to s3://{bucket}/{key}")
        
        try:
            # Combine splits and dividends
            combined = pd.concat([self.splits, self.dividends])
            
            if combined.empty:
                logger.warning(f"No corporate actions found for period {combined} ")
                # Upload empty dataframe to maintain schema
                combined = pd.DataFrame(columns=[
                    'execution_date', 'declaration_date', 'record_date', 
                    'payment_date', 'type', 'cash_amount', 'split_ratio'
                ])
            
            # Convert dates to datetime
            date_cols = ['execution_date', 'declaration_date', 'record_date', 'payment_date']
            combined[date_cols] = combined[date_cols].apply(pd.to_datetime, errors='coerce')
            
            # Add ticker column
            combined['ticker'] = ticker
            
            logger.info(f"📦 Uploading {len(combined)} corporate actions")
            success = upload_parquet_to_s3(combined, bucket, key)
            
            if success:
                logger.info(f"✅ Successfully uploaded to s3://{bucket}/{key}")
                return f"s3://{bucket}/{key}"
            return None

        except Exception as e:
            logger.error(f"🔥 Failed to upload corporate actions: {str(e)}")
            return None




# Singleton instance
corporate_actions_manager = corporate_actions_manager()
