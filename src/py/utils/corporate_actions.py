import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List
import logging

# Shared utilities
from src.py.utils.api_client import fetch_paginated_data

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
        """Fetch splits and dividends"""
        self._fetch_splits(symbols, start_date, end_date)
        self._fetch_dividends(symbols, start_date, end_date)
        self._create_adjustment_maps()

    def _fetch_splits(self, symbols: List[str], start_date: str, end_date: str):
        """Fetch stock splits"""
        url = f"{BASE_URL}/v3/reference/splits"
        params = {
            'ticker.in': ','.join(symbols),
            'execution_date.gte': start_date,
            'execution_date.lte': end_date,
            'limit': 1000
        }
        
        data = fetch_paginated_data(url, params)
        if data:
            self.splits = pd.DataFrame(data)[['ticker', 'execution_date', 'split_from', 'split_to']]
            self.splits.rename(columns={'ticker': 'symbol'}, inplace=True)

    def _fetch_dividends(self, symbols: List[str], start_date: str, end_date: str):
        """Fetch dividends"""
        url = f"{BASE_URL}/v3/reference/dividends"
        params = {
            'ticker.in': ','.join(symbols),
            'ex_dividend_date.gte': start_date,
            'ex_dividend_date.lte': end_date,
            'limit': 1000
        }
        
        data = fetch_paginated_data(url, params)
        if data:
            self.dividends = pd.DataFrame(data)[['ticker', 'ex_dividend_date', 'cash_amount']]
            self.dividends.rename(columns={'ticker': 'symbol'}, inplace=True)

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

# Singleton instance
corporate_actions_manager = corporate_actions_manager()
