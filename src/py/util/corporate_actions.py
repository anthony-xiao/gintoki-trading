import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from typing import List
import logging
from src.py.data_ingestion.historical_data_fetcher import upload_parquet_to_s3
from src.py.util.api_client import fetch_paginated_data

load_dotenv()
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
BASE_URL = "https://api.polygon.io"

class CorporateActionsManager:
    def __init__(self):
        self.splits = pd.DataFrame()
        self.dividends = pd.DataFrame()
        
    def fetch_market_actions(self, start_date: str, end_date: str):
        """Fetch all corporate actions for entire market"""
        self._fetch_all_splits(start_date, end_date)
        self._fetch_all_dividends(start_date, end_date)
        
    def _fetch_all_splits(self, start_date: str, end_date: str):
        """Fetch all stock splits in market"""
        url = f"{BASE_URL}/v3/reference/splits"
        params = {
            'execution_date.gte': start_date,
            'execution_date.lte': end_date,
            'limit': 1000
        }
        
        data = fetch_paginated_data(url, params)
        if data:
            self.splits = pd.DataFrame(data)[[
                'ticker', 'execution_date', 'split_from', 'split_to'
            ]].rename(columns={
                'ticker': 'symbol',
                'execution_date': 'date'
            })
            self.splits['split_ratio'] = self.splits['split_to'] / self.splits['split_from']
            
    def _fetch_all_dividends(self, start_date: str, end_date: str):
        """Fetch all dividends in market"""
        url = f"{BASE_URL}/v3/reference/dividends"
        params = {
            'ex_dividend_date.gte': start_date,
            'ex_dividend_date.lte': end_date,
            'limit': 1000
        }
        
        data = fetch_paginated_data(url, params)
        if data:
            self.dividends = pd.DataFrame(data)[[
                'ticker', 'ex_dividend_date', 'cash_amount', 'declaration_date'
            ]].rename(columns={
                'ticker': 'symbol',
                'ex_dividend_date': 'date'
            })

    def upload_market_actions(self, bucket: str, start: str, end: str):
        """Upload splits and dividends as market-wide datasets"""
        try:
            # Upload splits
            if not self.splits.empty:
                splits_key = f"corporate_actions/splits/{start}_to_{end}.parquet"
                if upload_parquet_to_s3(self.splits, bucket, splits_key):
                    logging.info(f"Uploaded market splits to s3://{bucket}/{splits_key}")
                    
            # Upload dividends
            if not self.dividends.empty:
                dividends_key = f"corporate_actions/dividends/{start}_to_{end}.parquet"
                if upload_parquet_to_s3(self.dividends, bucket, dividends_key):
                    logging.info(f"Uploaded market dividends to s3://{bucket}/{dividends_key}")
                    
            return True
        except Exception as e:
            logging.error(f"Failed to upload market corporate actions: {str(e)}")
            return False

# Singleton instance
corporate_actions_manager = CorporateActionsManager()
